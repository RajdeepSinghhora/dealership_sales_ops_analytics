#!/usr/bin/env python3
"""
validate_data.py  -  the quality gate that runs before any data is published.

These are PIPELINE checks (is the data technically sound?), not CRM hygiene checks.
Hygiene problems such as duplicate accounts are planted on purpose; finding those is the job
of your Power BI model. This script only makes sure the pipeline didn't break anything:

  schema        every file and required column exists
  keys          primary keys are unique and never blank
  references    every foreign key points to a real record (planted orphans excepted)
  consistency   stage history matches opportunity status, revenue matches product lines,
                converted leads point to an opportunity
  time          nothing is dated in the future, responses come after enquiries, activities
                come after the record they belong to
  freshness     data is as of yesterday and today's pipeline snapshot exists
  append-only   no table lost rows since the last run
  volume        the latest day's enquiry count looks normal

Results go to _validation_report.json (latest run) and _validation_log.csv (history of every
run, so you can chart pipeline health in Power BI). Exit code 1 if any check FAILS, which stops
the GitHub Action before bad data is published.

Usage
    python validate_data.py                   # freshness problems are warnings
    python validate_data.py --expect-fresh    # freshness problems fail (used in GitHub Actions)
"""

import argparse
import glob
import json
import os
import sys
import time

import pandas as pd

REQUIRED = {
    "dim_outlets.csv": ["OutletID", "OutletName", "City", "Zone", "Channel"],
    "dim_users.csv": ["UserID", "FullName", "Role", "OutletID", "ManagerUserID", "JoinDate", "ExitDate", "IsActive"],
    "dim_products.csv": ["ProductID", "Model", "FuelType", "IsEV", "LaunchDate", "IndicativeExShowroomINR"],
    "dim_accounts.csv": ["AccountID", "AccountName", "AccountType", "ParentAccountID", "GSTIN", "DUNS", "ParentDUNS",
                         "KN_Designation", "OwnerUserID", "CreatedOn"],
    "dim_contacts.csv": ["ContactID", "CustomerType", "AccountID", "FullName", "Mobile", "Email", "CreatedOn"],
    "fact_leads.csv": ["LeadID", "CreatedOn", "Channel", "Source", "OutletID", "OwnerUserID", "AccountID",
                       "InterestedProductID", "Status", "FirstResponseOn", "ConvertedOn", "OpportunityID"],
    "fact_opportunities.csv": ["OpportunityID", "LeadID", "Channel", "OutletID", "AccountID", "ContactID",
                               "OwnerUserID", "CreatedOn", "StageNo", "Stage", "Status", "TotalUnits",
                               "EstimatedRevenueINR", "ExpectedCloseDate", "ActualCloseDate", "LastActivityOn"],
    "fact_opportunity_lines.csv": ["LineID", "OpportunityID", "ProductID", "Quantity", "LineValueINR"],
    "fact_stage_history.csv": ["StageHistoryID", "OpportunityID", "StageNo", "Stage", "EnteredOn", "ExitedOn"],
    "fact_targets.csv": ["MonthStart", "OutletID", "Channel", "TargetUnits", "TargetRevenueINR"],
    "fact_report_requests.csv": ["RequestID", "RequestedOn", "Priority", "DueOn", "DeliveredOn", "Status",
                                 "AssignedToUserID"],
    "fact_pipeline_snapshot.csv": ["SnapshotDate", "OutletID", "Channel", "StageNo", "OpenDeals", "OpenValueINR"],
    "_data_quality_answer_key.csv": ["IssueType", "Table", "RecordID", "RelatedRecordID"],
}
ACT_COLS = ["ActivityID", "ActivityOn", "RegardingType", "RegardingID", "OwnerUserID", "OutletID"]
GROW_ONLY = ["dim_users.csv", "dim_accounts.csv", "dim_contacts.csv", "fact_leads.csv", "fact_opportunities.csv",
             "fact_opportunity_lines.csv", "fact_stage_history.csv", "fact_report_requests.csv",
             "fact_pipeline_snapshot.csv"]


class Validator:
    def __init__(self, data, expect_fresh):
        self.data, self.expect_fresh = data, expect_fresh
        self.results = []

    def rec(self, check, ok, detail="", warn_only=False):
        status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
        self.results.append(dict(Check=check, Status=status, Detail="" if ok else detail))
        mark = {"PASS": "  ok ", "WARN": " warn", "FAIL": " FAIL"}[status]
        print(f"{mark}  {check}" + (f"  ->  {detail}" if detail and status != "PASS" else ""))

    def load(self):
        self.t = {}
        missing = []
        for f, cols in REQUIRED.items():
            path = os.path.join(self.data, f)
            if not os.path.exists(path):
                missing.append(f)
                continue
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            lacking = [c for c in cols if c not in df.columns]
            if lacking:
                missing.append(f"{f}: {', '.join(lacking)}")
            self.t[f] = df
        files = sorted(glob.glob(os.path.join(self.data, "activities", "fact_activities_FY*.csv")))
        if not files:
            missing.append("activities/fact_activities_FY*.csv")
        else:
            self.t["activities"] = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=ACT_COLS)
                                              for f in files], ignore_index=True)
        self.rec("schema: files and required columns present", not missing, "; ".join(missing))
        with open(os.path.join(self.data, "_generation_summary.json")) as fh:
            self.summary = json.load(fh)
        self.as_of = pd.Timestamp(self.summary["data_as_of"])
        return not missing

    # ------------------------------------------------------------------ checks
    def keys(self):
        pks = {"dim_outlets.csv": "OutletID", "dim_users.csv": "UserID", "dim_products.csv": "ProductID",
               "dim_accounts.csv": "AccountID", "dim_contacts.csv": "ContactID", "fact_leads.csv": "LeadID",
               "fact_opportunities.csv": "OpportunityID", "fact_opportunity_lines.csv": "LineID",
               "fact_stage_history.csv": "StageHistoryID", "fact_report_requests.csv": "RequestID",
               "activities": "ActivityID"}
        for f, k in pks.items():
            s = self.t[f][k]
            blanks, dups = int((s == "").sum()), int(s.duplicated().sum())
            self.rec(f"keys: {f}.{k} unique and not blank", blanks == 0 and dups == 0,
                     f"{blanks} blank, {dups} duplicated")

    def references(self):
        T = self.t
        ids = {f: set(T[f][k]) for f, k in [("dim_accounts.csv", "AccountID"), ("dim_contacts.csv", "ContactID"),
                                              ("dim_users.csv", "UserID"), ("fact_leads.csv", "LeadID"),
                                              ("fact_opportunities.csv", "OpportunityID"),
                                              ("dim_outlets.csv", "OutletID"), ("dim_products.csv", "ProductID")]}
        key = T["_data_quality_answer_key.csv"]
        planted_orphans = set(key.loc[key["IssueType"] == "orphan_parent_reference", "RecordID"])

        def fk(label, s, target, allowed_rows=None):
            s = s[s != ""]
            bad = s[~s.isin(ids[target])]
            if allowed_rows is not None:
                bad = bad[~allowed_rows.reindex(bad.index).fillna(False)]
            self.rec(f"references: {label}", len(bad) == 0, f"{len(bad)} orphan(s), e.g. {list(bad.head(3))}")

        o, l, a = T["fact_opportunities.csv"], T["fact_leads.csv"], T["dim_accounts.csv"]
        fk("opportunity -> account", o["AccountID"], "dim_accounts.csv")
        fk("opportunity -> contact", o["ContactID"], "dim_contacts.csv")
        fk("opportunity -> owner", o["OwnerUserID"], "dim_users.csv")
        fk("opportunity -> lead", o["LeadID"], "fact_leads.csv")
        fk("opportunity -> outlet", o["OutletID"], "dim_outlets.csv")
        fk("lead -> opportunity", l["OpportunityID"], "fact_opportunities.csv")
        fk("lead -> account", l["AccountID"], "dim_accounts.csv")
        fk("lead -> owner", l["OwnerUserID"], "dim_users.csv")
        fk("lead -> product", l["InterestedProductID"], "dim_products.csv")
        fk("contact -> account", T["dim_contacts.csv"]["AccountID"], "dim_accounts.csv")
        fk("account -> owner", a["OwnerUserID"], "dim_users.csv")
        fk("account -> parent (planted orphans allowed)", a["ParentAccountID"], "dim_accounts.csv",
           a["AccountID"].isin(planted_orphans))
        fk("line -> opportunity", T["fact_opportunity_lines.csv"]["OpportunityID"], "fact_opportunities.csv")
        fk("line -> product", T["fact_opportunity_lines.csv"]["ProductID"], "dim_products.csv")
        fk("stage history -> opportunity", T["fact_stage_history.csv"]["OpportunityID"], "fact_opportunities.csv")
        act = T["activities"]
        fk("activity -> lead", act.loc[act["RegardingType"] == "Lead", "RegardingID"], "fact_leads.csv")
        fk("activity -> opportunity", act.loc[act["RegardingType"] == "Opportunity", "RegardingID"],
           "fact_opportunities.csv")
        fk("activity -> owner", act["OwnerUserID"], "dim_users.csv")
        fk("report request -> analyst", T["fact_report_requests.csv"]["AssignedToUserID"], "dim_users.csv")

    def consistency(self):
        T = self.t
        o, h = T["fact_opportunities.csv"], T["fact_stage_history.csv"]
        expected = o["Status"].map({"Open": None, "Won": "6", "Lost": "7"})
        bad_status = o[((o["Status"] == "Open") & ~o["StageNo"].isin(list("12345"))) |
                       ((o["Status"] != "Open") & (o["StageNo"] != expected))]
        self.rec("consistency: opportunity status matches stage", len(bad_status) == 0, f"{len(bad_status)} mismatched")

        open_rows = h[(h["ExitedOn"] == "") & h["StageNo"].isin(list("12345"))]
        per_opp = open_rows.groupby("OpportunityID")["StageNo"].agg(["count", "first"])
        op = o[o["Status"] == "Open"].set_index("OpportunityID")
        chk = op.join(per_opp, how="left")
        bad = chk[(chk["count"] != 1) | (chk["first"] != chk["StageNo"])]
        extra = per_opp.index.difference(op.index)
        self.rec("consistency: each open deal has exactly one open stage row, at its current stage",
                 len(bad) == 0 and len(extra) == 0, f"{len(bad)} open deals wrong, {len(extra)} closed deals with open rows")

        last = h.sort_values(["OpportunityID", "EnteredOn", "StageNo"]).groupby("OpportunityID")["StageNo"].last()
        cl = o[o["Status"] != "Open"].set_index("OpportunityID")
        bad = cl[last.reindex(cl.index) != cl["StageNo"]]
        no_date = cl[cl["ActualCloseDate"] == ""]
        self.rec("consistency: closed deals end in their closing stage and have a close date",
                 len(bad) == 0 and len(no_date) == 0, f"{len(bad)} wrong last stage, {len(no_date)} missing close date")

        ln = T["fact_opportunity_lines.csv"]
        sums = pd.to_numeric(ln["LineValueINR"]).groupby(ln["OpportunityID"]).sum()
        rev = pd.to_numeric(o.set_index("OpportunityID")["EstimatedRevenueINR"])
        diff = (rev - sums.reindex(rev.index).fillna(0)).abs()
        bad = diff[diff > 1]
        self.rec("consistency: opportunity revenue equals the sum of its product lines", len(bad) == 0,
                 f"{len(bad)} deals differ, e.g. {list(bad.index[:3])}")

        l = T["fact_leads.csv"]
        bad = l[(l["Status"] == "Converted") != (l["OpportunityID"] != "")]
        self.rec("consistency: converted leads (and only those) point to an opportunity", len(bad) == 0,
                 f"{len(bad)} leads inconsistent")

    def timing(self):
        T = self.t
        limit = (self.as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        cols = [("fact_leads.csv", ["CreatedOn", "FirstResponseOn", "LastContactedOn", "ConvertedOn"]),
                ("fact_opportunities.csv", ["CreatedOn", "LastActivityOn"]),
                ("fact_stage_history.csv", ["EnteredOn", "ExitedOn"]), ("dim_accounts.csv", ["CreatedOn"]),
                ("dim_contacts.csv", ["CreatedOn"]), ("activities", ["ActivityOn"]),
                ("fact_report_requests.csv", ["RequestedOn", "DeliveredOn"])]
        future = []
        for f, cs in cols:
            for c in cs:
                n = int((T[f][c] >= limit).sum())
                if n:
                    future.append(f"{f}.{c}: {n}")
        self.rec("time: nothing dated after the as-of date", not future, "; ".join(future))

        l = T["fact_leads.csv"]
        bad_fr = l[(l["FirstResponseOn"] != "") & (l["FirstResponseOn"] < l["CreatedOn"])]
        bad_cv = l[(l["ConvertedOn"] != "") & (l["ConvertedOn"] < l["CreatedOn"])]
        self.rec("time: lead responses and conversions come after the enquiry", len(bad_fr) + len(bad_cv) == 0,
                 f"{len(bad_fr)} early responses, {len(bad_cv)} early conversions")

        h = T["fact_stage_history.csv"]
        bad = h[(h["ExitedOn"] != "") & (h["ExitedOn"] < h["EnteredOn"])]
        self.rec("time: stage exits come after stage entries", len(bad) == 0, f"{len(bad)} rows")

        act = T["activities"]
        ao = act[act["RegardingType"] == "Opportunity"]
        created = ao["RegardingID"].map(T["fact_opportunities.csv"].set_index("OpportunityID")["CreatedOn"])
        bad = ao[ao["ActivityOn"] < created]
        self.rec("time: activities come after the deal they belong to was created", len(bad) == 0, f"{len(bad)} rows")

    def freshness(self):
        yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        self.rec("freshness: data is as of yesterday", self.as_of == yesterday,
                 f"data as of {self.as_of.date()}, expected {yesterday.date()}", warn_only=not self.expect_fresh)
        s = self.t["fact_pipeline_snapshot.csv"]
        self.rec("freshness: pipeline snapshot exists for the as-of date",
                 (s["SnapshotDate"] == self.as_of.strftime("%Y-%m-%d")).any(),
                 f"no snapshot rows for {self.as_of.date()}")

    def append_only(self):
        prev = self.summary.get("previous_row_counts") or {}
        now = self.summary.get("row_counts") or {}
        shrunk = []
        for f in GROW_ONLY + [k for k in now if k.startswith("activities/")]:
            if f in prev and f in now and int(now[f]) < int(prev[f]):
                shrunk.append(f"{f}: {prev[f]} -> {now[f]}")
        self.rec("append-only: no table lost rows since the last run", not shrunk, "; ".join(shrunk))

    def volume(self):
        l = self.t["fact_leads.csv"]
        n = int((l["CreatedOn"].str[:10] == self.as_of.strftime("%Y-%m-%d")).sum())
        scale = float(self.summary.get("scale", 1))
        lo, hi = 20 * scale, 400 * scale
        self.rec("volume: enquiries on the latest day look normal", lo <= n <= hi,
                 f"{n} enquiries on {self.as_of.date()} (expected {lo:.0f}-{hi:.0f})", warn_only=True)

    # ------------------------------------------------------------------ run
    def run(self):
        print(f"Validating {os.path.abspath(self.data)}")
        if self.load():
            for step in (self.keys, self.references, self.consistency, self.timing, self.freshness,
                         self.append_only, self.volume):
                try:
                    step()
                except Exception as e:                           # a crashing check is a failing check
                    self.rec(f"{step.__name__}: check crashed", False, repr(e))
        fails = sum(r["Status"] == "FAIL" for r in self.results)
        warns = sum(r["Status"] == "WARN" for r in self.results)
        overall = "FAIL" if fails else ("PASS with warnings" if warns else "PASS")
        run_at = time.strftime("%Y-%m-%d %H:%M:%S")
        as_of = self.summary["data_as_of"] if hasattr(self, "summary") else ""
        with open(os.path.join(self.data, "_validation_report.json"), "w") as f:
            json.dump(dict(run_at=run_at, data_as_of=as_of, overall=overall, failed=fails, warnings=warns,
                           checks=self.results), f, indent=2)
        log = pd.DataFrame(self.results)
        log.insert(0, "DataAsOf", as_of)
        log.insert(0, "RunAt", run_at)
        path = os.path.join(self.data, "_validation_log.csv")
        log.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"\nOverall: {overall}  ({len(self.results)} checks, {fails} failed, {warns} warnings)")
        return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate the CRM data before publishing")
    ap.add_argument("--data", default="data")
    ap.add_argument("--expect-fresh", action="store_true", help="fail (not warn) if data is not as of yesterday")
    a = ap.parse_args()
    sys.exit(Validator(a.data, a.expect_fresh).run())