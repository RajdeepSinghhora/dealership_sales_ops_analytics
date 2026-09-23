#!/usr/bin/env python3
"""
daily_update.py  -  Phase 2 of the "Dealership Sales Ops Command Center" portfolio project

Moves the synthetic Anvaya Motors CRM forward one day at a time, from the last as-of date up to
yesterday. For every day it:
  * adds new retail and corporate enquiries (seasonality, weekends, growth)
  * works open leads: first response, follow-ups, conversion, lost / unresponsive
  * moves open opportunities through stages, closes them as won / lost, logs activities
  * creates new accounts and contacts when new companies convert
  * simulates staff leaving (some of their records are never reassigned)
  * logs new reporting requests (SLA tracking) and sets targets for each new month
  * plants fresh data-quality problems of the same kinds as Phase 1
  * appends one row per outlet / channel / stage to fact_pipeline_snapshot.csv
    (on the first run the snapshot is backfilled for the whole history from stage history)
Then it refreshes the answer key, the manifest (_manifest.csv) and _generation_summary.json.

Usage
    python daily_update.py                        # catch up to yesterday
    python daily_update.py --until 2026-10-05     # catch up to a specific date (testing)
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import generate_history as gh

FMT = "%Y-%m-%d %H:%M:%S"
DAY = pd.Timedelta(days=1)
OPEN_LEAD = {"New", "Contacted", "Qualified"}
RULE_BASED = {"key_account_missing_duns", "account_owner_inactive", "open_opportunity_owner_inactive",
              "stale_open_opportunity", "abandoned_open_opportunity", "open_opportunity_past_expected_close",
              "unconverted_lead_over_30_days"}
TENURE_MONTHS = {"Sales Consultant": 16, "Sales Team Lead": 30, "Outlet Sales Manager": 40,
                 "Corporate Sales Executive": 22, "Key Account Manager": 36,
                 "Head - Corporate & Fleet Sales": 60, "Sales Ops / MIS Analyst": 26, "GM - Sales": 70}
TARGET_RANGE = {"Sales Consultant": (8, 15), "Corporate Sales Executive": (15, 45), "Key Account Manager": (40, 90)}
OUTLET_W = {o[0]: o[5] for o in gh.OUTLETS}
OUTLET_CITY = {o[0]: o[2] for o in gh.OUTLETS}
RETAIL_ACTS = (["Call", "WhatsApp", "Showroom Visit", "Test Drive", "Email", "Quotation Sent", "SMS Follow-up"],
               [35, 30, 10, 6, 7, 5, 7])
B2B_ACTS = (["Call", "Email", "Meeting", "WhatsApp", "Product Demo", "Site Visit", "Quotation Sent"],
            [30, 25, 15, 15, 5, 5, 5])
DUR_MEAN = {"Call": 4, "Meeting": 45, "Site Visit": 60, "Test Drive": 30, "Showroom Visit": 40, "Product Demo": 45,
            "Vehicle Delivery": 60}
STAGE_ACT = {1: ("Test Drive", "Product Demo"), 2: ("Quotation Sent", "Quotation Sent"),
             3: ("Showroom Visit", "Meeting"), 4: ("Call", "Meeting"), 5: ("Vehicle Delivery", "Meeting")}
SLA_H = {"P1": 4, "P2": 24, "P3": 72}
REQ_CATS = {"Daily Sales Flash": ("Standing - Daily", "P1", 10), "Weekly Pipeline Review": ("Standing - Weekly", "P2", 8),
            "Target vs Achievement": ("Standing - Monthly", "P2", 6), "Incentive Calculation": ("Standing - Monthly", "P1", 4),
            "OEM Submission": ("Standing - Monthly", "P1", 4), "Lead Source ROI": ("Ad hoc", "P3", 5),
            "Data Correction": ("Ad hoc", "P2", 9), "Ad hoc Analysis": ("Ad hoc", "P3", 12),
            "Board / Owner Review Pack": ("Ad hoc", "P1", 2)}
SNAP_COLS = ["SnapshotDate", "OutletID", "Channel", "StageNo", "Stage", "OpenDeals", "OpenUnits", "OpenValueINR",
             "StaleDeals", "OverdueDeals"]


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def ts(x):
    return x.strftime(FMT)


class Helper(gh.Generator):
    """Re-uses the Phase 1 helpers (names, GSTIN, products...) without building a full history."""

    def __init__(self, start, used_names, used_duns, scale):
        self.START, self.scale, self.issues = start, scale, []
        self.used_names, self.used_duns = used_names, used_duns
        self.rng, self.T = np.random.default_rng(0), 1
        self.build_products()


class Updater:
    def __init__(self, data):
        self.data = data
        with open(self.p("_generation_summary.json")) as f:
            self.summary = json.load(f)
        s = self.summary
        self.START, self.as_of = pd.Timestamp(s["data_start"]), pd.Timestamp(s["data_as_of"])
        self.scale, self.seed = float(s["scale"]), int(s["seed"])
        self.nx = dict(s["next_ids"])
        self.prev_counts = dict(s.get("row_counts", {}))

        self.users = read(self.p("dim_users.csv")).to_dict("records")
        self.accounts = read(self.p("dim_accounts.csv")).to_dict("records")
        self.contacts = read(self.p("dim_contacts.csv")).to_dict("records")
        self.leads = read(self.p("fact_leads.csv")).to_dict("records")
        self.opps = read(self.p("fact_opportunities.csv")).to_dict("records")
        self.hist = read(self.p("fact_stage_history.csv")).to_dict("records")
        self.requests = read(self.p("fact_report_requests.csv")).to_dict("records")
        self.targets = read(self.p("fact_targets.csv"))
        self.key = read(self.p("_data_quality_answer_key.csv"))
        self.cols = {n: list(pd.read_csv(self.p(n), nrows=0).columns) for n in
                     ["dim_users.csv", "dim_accounts.csv", "dim_contacts.csv", "fact_leads.csv",
                      "fact_opportunities.csv", "fact_stage_history.csv", "fact_report_requests.csv",
                      "fact_opportunity_lines.csv"]}
        lines = read(self.p("fact_opportunity_lines.csv"))
        self.first_prod = lines.drop_duplicates("OpportunityID").set_index("OpportunityID")["ProductID"].to_dict()

        self.h = Helper(self.START, {a["AccountName"] for a in self.accounts},
                        {a["DUNS"] for a in self.accounts if a["DUNS"]}, self.scale)
        self.issues, self.new_lines, self.snap_rows = [], [], []
        self.acc_by_id = {a["AccountID"]: a for a in self.accounts}
        self.contacts_by_acc = {}
        for c in self.contacts:
            if c["AccountID"]:
                self.contacts_by_acc.setdefault(c["AccountID"], []).append(c)
        dups = self.key[self.key["IssueType"] == "duplicate_account"]
        self.dup_ids = set(dups["RecordID"])
        self.orig_to_dup = dict(zip(dups["RelatedRecordID"], dups["RecordID"]))
        unl = self.key[self.key["IssueType"] == "b2b_lead_not_linked_to_existing_account"]
        self.unlinked = dict(zip(unl["RecordID"], unl["RelatedRecordID"]))

        self.open_leads = [L for L in self.leads if L["Status"] in OPEN_LEAD]
        for L in self.open_leads:
            L["_created"] = pd.Timestamp(L["CreatedOn"])
            L["_fr"] = pd.Timestamp(L["FirstResponseOn"]) if L["FirstResponseOn"] else None
        self.open_hist = {}
        for hrow in self.hist:
            if hrow["ExitedOn"] == "" and int(hrow["StageNo"]) <= 5:
                hrow["_entered"] = pd.Timestamp(hrow["EnteredOn"])
                self.open_hist[hrow["OpportunityID"]] = hrow
        self.open_opps = [o for o in self.opps if o["Status"] == "Open"]
        for o in self.open_opps:
            self.prep_opp(o)
        self.next_hist = len(self.hist) + 1
        self.next_user = max(int(u["UserID"][4:]) for u in self.users) + 1

    # ------------------------------------------------------------------ small helpers
    def p(self, *parts):
        return os.path.join(self.data, *parts)

    def tf(self, when):
        return (when - self.START) / DAY

    def nid(self, key, prefix, width):
        v = int(self.nx[key])
        self.nx[key] = v + 1
        return f"{prefix}-{v:0{width}d}"

    def issue(self, itype, table, rid, related="", detail=""):
        self.issues.append(dict(IssueType=itype, Table=table, RecordID=rid, RelatedRecordID=related, Detail=detail))

    def at(self, lo, hi):
        """Random timestamp today between hours lo and hi."""
        return self.day + pd.Timedelta(hours=float(self.rng.uniform(lo, hi)))

    def prep_opp(self, o):
        o["_created"] = pd.Timestamp(o["CreatedOn"])
        o["_last"] = pd.Timestamp(o["LastActivityOn"]) if o["LastActivityOn"] else o["_created"]
        o["_retail"] = o["Channel"] == "Retail"
        o["_tender"] = o["IsTender"] == "True"
        o["_ev"] = self.h.prod_ev.get(self.first_prod.get(o["OpportunityID"], ""), False)

    def pool(self, role, outlet=None):
        return self.pools.get((role, outlet), []) if outlet else self.pools.get((role, None), [])

    def build_pools(self):
        ds = self.day.strftime("%Y-%m-%d")
        self.pools = {}
        for u in self.users:
            if u["ExitDate"] == "" and u["JoinDate"] <= ds:
                self.pools.setdefault((u["Role"], u["OutletID"]), []).append(u["UserID"])
                self.pools.setdefault((u["Role"], None), []).append(u["UserID"])
        self.active = {uid for v in self.pools.values() for uid in v}

    def choose(self, *pools):
        for pl in pools:
            if pl:
                return pl[int(self.rng.integers(0, len(pl)))]
        return ""

    def retail_owner(self, outlet):
        return self.choose(self.pool("Sales Consultant", outlet), self.pool("Sales Team Lead", outlet),
                           self.pool("Outlet Sales Manager", outlet))

    def b2b_owner(self, key_account=False):
        first = self.pool("Key Account Manager") if key_account else self.pool("Corporate Sales Executive")
        return self.choose(first, self.pool("Corporate Sales Executive"), self.pool("Head - Corporate & Fleet Sales"))

    def act(self, rtype, rid, owner, outlet, channel, when, atype=None, outcome=None):
        r = self.rng
        if atype is None:
            names, w = RETAIL_ACTS if channel == "Retail" else B2B_ACTS
            atype = names[r.choice(len(names), p=np.array(w) / sum(w))]
        if outcome is None:
            if atype == "Call":
                outcome = ["Connected", "No Answer", "Callback Requested"][r.choice(3, p=[.55, .3, .15])]
            elif atype in ("WhatsApp", "SMS Follow-up", "Email"):
                outcome = ["Delivered", "Replied"][r.choice(2, p=[.65, .35])]
            else:
                outcome = "Completed"
        dur = 0.0 if outcome == "No Answer" else (round(float(r.gamma(2.0, DUR_MEAN[atype] / 2.0)), 1)
                                                   if atype in DUR_MEAN else 0.0)
        when = min(when, self.day_end - pd.Timedelta(seconds=1))
        self.acts.append(dict(ActivityOn=when, RegardingType=rtype, RegardingID=rid, OwnerUserID=owner,
                              OutletID=outlet, Channel=channel, ActivityType=atype, Outcome=outcome, DurationMins=dur))
        return when

    # ------------------------------------------------------------------ people
    def attrition(self):
        r, ds = self.rng, self.day.strftime("%Y-%m-%d")
        for u in list(self.users):
            if u["ExitDate"] or u["JoinDate"] > ds or u["Role"] not in TENURE_MONTHS:
                continue
            if r.random() >= 1 / (TENURE_MONTHS[u["Role"]] * 30.44):
                continue
            u["ExitDate"] = ds
            nm = self.h.names(1)[0]
            first, last = nm.lower().replace("'", "").split(" ", 1)
            uid = f"USR-{self.next_user:03d}"
            self.next_user += 1
            tr = TARGET_RANGE.get(u["Role"])
            self.users.append(dict(UserID=uid, FullName=nm, Role=u["Role"], Channel=u["Channel"], OutletID=u["OutletID"],
                                   ManagerUserID=u["ManagerUserID"],
                                   Email=f"{first}.{last.replace(' ', '')}{uid[4:]}@{gh.STAFF_EMAIL_DOMAIN}",
                                   JoinDate=(self.day + int(r.integers(3, 31)) * DAY).strftime("%Y-%m-%d"),
                                   ExitDate="", IsActive="False",
                                   MonthlyUnitTarget=str(int(r.integers(*tr))) if tr else ""))
            # the manager reassigns part of the leaver's book to colleagues; the rest is left behind
            colleagues = [x for x in self.pool(u["Role"], u["OutletID"]) if x != u["UserID"]]
            if not colleagues:
                continue
            for a in self.accounts:
                if a["OwnerUserID"] == u["UserID"] and r.random() < 0.75:
                    a["OwnerUserID"] = self.choose(colleagues)
            for rec in self.open_opps + self.open_leads:
                if rec["OwnerUserID"] == u["UserID"] and r.random() < 0.5:
                    rec["OwnerUserID"] = self.choose(colleagues)

    # ------------------------------------------------------------------ targets
    def monthly_targets(self):
        ms = self.day.replace(day=1).strftime("%Y-%m-%d")
        if (self.targets["MonthStart"] == ms).any():
            return
        lo = (self.day - 365 * DAY).strftime("%Y-%m-%d")
        won = [o for o in self.opps if o["Status"] == "Won" and o["ActualCloseDate"] >= lo]
        rows = []
        for oid in [o[0] for o in gh.OUTLETS if o[4] in ("Retail", "Corporate & Fleet")]:
            w = [o for o in won if o["OutletID"] == oid]
            units = sum(int(o["TotalUnits"] or 0) for o in w)
            rev = sum(int(o["EstimatedRevenueINR"] or 0) for o in w)
            mu, aup = units / 12, rev / max(units, 1)
            tu = max(1, int(round(mu * gh.SEASON[self.day.month] * self.rng.uniform(1.0, 1.22))))
            rows.append(dict(MonthStart=ms, OutletID=oid, Channel="Retail" if oid in gh.RETAIL_OUTLETS else
                             "Corporate & Fleet", TargetUnits=str(tu), TargetRevenueINR=str(int(round(tu * aup, -3)))))
        self.targets = pd.concat([self.targets, pd.DataFrame(rows)], ignore_index=True)

    # ------------------------------------------------------------------ new leads
    def new_lead(self, **kw):
        L = dict(LeadID=self.nid("lead", "LD", 7), CreatedOn="", Channel="", Source="", OutletID="", OwnerUserID="",
                 LeadName="", CompanyName="", AccountID="", Mobile="", Email="", City="", InterestedProductID="",
                 IsExistingCustomer="False", Status="New", FirstResponseOn="", LastContactedOn="", ConvertedOn="",
                 OpportunityID="")
        L.update(kw)
        L["CreatedOn"] = ts(L["_created"])
        if L.get("_fr") is not None:
            L.update(FirstResponseOn=ts(L["_fr"]), LastContactedOn=ts(L["_fr"]), Status="Contacted")
        self.leads.append(L)
        self.open_leads.append(L)
        return L

    def new_retail_leads(self):
        r, D = self.rng, self.day
        yrs = (D - self.START).days / 365.25
        hour_w = np.array([1, .5, .3, .2, .2, .3, .6, 1.2, 2, 3, 3.5, 4, 4, 3.5, 3.5, 3.5, 3.5, 4, 4.5, 5, 5, 4.5,
                           3.5, 2])
        srcs, sw = list(gh.RETAIL_SOURCES), np.array(list(gh.RETAIL_SOURCES.values()))
        todays = []
        for oid in gh.RETAIL_OUTLETS:
            lam = 11.0 * self.scale * OUTLET_W[oid] * gh.SEASON[D.month] * gh.WEEKDAY[D.dayofweek] * (1 + 0.07 * yrs)
            for _ in range(r.poisson(lam)):
                src = srcs[r.choice(len(srcs), p=sw / sw.sum())]
                online = src in gh.ONLINE_SOURCES
                hour = r.choice(24, p=hour_w / hour_w.sum()) + r.random() if online else r.uniform(10, 19.5)
                created = D + pd.Timedelta(hours=float(hour))
                if src in ("Walk-in", "Exchange Mela / Event"):
                    fr = created
                elif src == "Referral":
                    fr = created + pd.Timedelta(minutes=float(r.lognormal(np.log(90), 0.9)))
                elif 9.5 <= hour < 19:
                    fr = created + pd.Timedelta(minutes=float(r.lognormal(np.log(30), 1.0)))
                elif hour < 9.5:
                    fr = D + pd.Timedelta(hours=9.5) + pd.Timedelta(minutes=float(r.lognormal(np.log(30), 0.8)))
                else:
                    fr = None                                   # arrived after hours: handled tomorrow
                if online and r.random() < 0.05:
                    fr = None
                if fr is not None and fr >= self.day_end:
                    fr = None
                owner = self.retail_owner(oid)
                nm, mob = self.h.names(1)[0], self.h.mobiles(1)[0]
                first, last = nm.lower().replace("'", "").split(" ", 1)
                email = f"{first}.{last.replace(' ', '')}{r.integers(1, 999)}@{self.h.pick(gh.PERSONAL_EMAIL_DOMAINS)}" \
                    if r.random() < 0.45 else ""
                prod = self.h.retail_products([self.tf(created)])[0]
                L = self.new_lead(_created=created, _fr=fr, Channel="Retail", Source=src, OutletID=oid,
                                  OwnerUserID=owner, LeadName=nm, Mobile=mob, Email=email, City=OUTLET_CITY[oid],
                                  InterestedProductID=prod)
                if online and r.random() < 0.015:
                    L["OwnerUserID"] = ""
                    self.issue("lead_unassigned", "fact_leads", L["LeadID"])
                if fr is not None:
                    self.act("Lead", L["LeadID"], L["OwnerUserID"], oid, "Retail", fr,
                             "Showroom Visit" if src == "Walk-in" else ("Call" if r.random() < .6 else "WhatsApp"))
                todays.append(L)
        # planted: the same customer enquires again through another source
        recent = [L for L in self.leads[-4000:] if L["Channel"] == "Retail" and L["CreatedOn"] >= ts(D - 10 * DAY)]
        for _ in range(r.binomial(len(todays), 0.03)):
            if not recent:
                break
            o = recent[int(r.integers(0, len(recent)))]
            created = self.at(10, 19.5)
            if created <= pd.Timestamp(o["CreatedOn"]):
                continue
            fr = min(created + pd.Timedelta(minutes=float(r.lognormal(np.log(45), 1.0))), self.day_end - pd.Timedelta(seconds=1))
            d = self.new_lead(_created=created, _fr=fr, Channel="Retail", OutletID=o["OutletID"],
                              Source=self.h.pick([s for s in gh.RETAIL_SOURCES if s != o["Source"]]),
                              OwnerUserID=self.retail_owner(o["OutletID"]), LeadName=o["LeadName"], Mobile=o["Mobile"],
                              Email=o["Email"], City=o["City"], InterestedProductID=o["InterestedProductID"])
            self.issue("duplicate_lead", "fact_leads", d["LeadID"], o["LeadID"],
                       "Same customer enquired again via another source")

    def new_b2b_leads(self):
        r, D = self.rng, self.day
        ds = D.strftime("%Y-%m-%d")
        elig = [a for a in self.accounts if a["AccountID"] not in self.dup_ids and a["CreatedOn"][:10] < ds]
        mult = {"Key": 2.0, "Named": 1.4}
        rates = np.array([gh.ACCT_TYPES[a["AccountType"]]["repeat"] * mult.get(a["KN_Designation"], 1.0) / 365.25
                          for a in elig])
        k = r.poisson(rates.sum())
        for i in (r.choice(len(elig), size=k, p=rates / rates.sum()) if k else []):
            a = elig[int(i)]
            src = self.h.pick(["Existing Customer", "Referral", "OEM Fleet Desk Referral", "Tender / RFP"],
                              p=[0.6, 0.15, 0.15, 0.10 if a["AccountType"] != "Government / PSU" else 0.5])
            dup = self.orig_to_dup.get(a["AccountID"])
            if dup and self.acc_by_id[dup]["CreatedOn"] < ts(D) and r.random() < 0.35:
                a = self.acc_by_id[dup]                          # deal logged on the duplicate record
            self.make_b2b_lead(a, src, repeat=True)
        # enquiries from companies that are not customers yet
        existing = [a for a in elig]
        for _ in range(r.poisson(3.0 * self.scale)):
            src = self.h.pick(list(gh.NEW_B2B_SOURCES), p=list(gh.NEW_B2B_SOURCES.values()))
            atype = self.h.pick(list(gh.ACCT_TYPES), p=[v["share"] for v in gh.ACCT_TYPES.values()])
            if atype == "Government / PSU":
                src = "Tender / RFP"
            if r.random() < 0.06 and existing:
                o = existing[int(r.integers(0, len(existing)))]
                L = self.make_b2b_lead(None, src, company=self.h.name_variant(o["AccountName"]), atype=o["AccountType"],
                                       city=o["City"])
                self.unlinked[L["LeadID"]] = o["AccountID"]
                self.issue("b2b_lead_not_linked_to_existing_account", "fact_leads", L["LeadID"], o["AccountID"],
                           f"Company '{L['CompanyName']}' already exists as an account")
            else:
                nm = self.h.new_account_name(atype)["name"]
                self.h.used_names.discard(nm)                    # only reserved once it becomes an account
                self.make_b2b_lead(None, src, company=nm, atype=atype,
                                   city=self.h.pick(gh.KA_CITIES, p=gh.KA_CITY_W))

    def make_b2b_lead(self, a, src, repeat=False, company="", atype=None, city=""):
        r = self.rng
        created = self.at(9.5, 18.5)
        fr = created + pd.Timedelta(hours=float(r.lognormal(np.log(6), 0.9)))
        fr = fr if fr < self.day_end else None
        if a is not None and a["OwnerUserID"] in self.active:
            owner = a["OwnerUserID"]
        else:
            owner = self.b2b_owner(a is not None and a["KN_Designation"] == "Key")
        atype = a["AccountType"] if a is not None else atype
        L = self.new_lead(_created=created, _fr=fr, Channel="Corporate & Fleet", Source=src, OutletID="OUT08",
                          OwnerUserID=owner, CompanyName=a["AccountName"] if a is not None else company,
                          AccountID=a["AccountID"] if a is not None else "",
                          City=a["City"] if a is not None else city,
                          InterestedProductID=self.h.b2b_product(atype, self.tf(created)),
                          IsExistingCustomer=str(bool(repeat)), _atype=atype)
        if fr is not None:
            self.act("Lead", L["LeadID"], owner, "OUT08", L["Channel"], fr, "Call" if r.random() < .6 else "Email")
        return L

    # ------------------------------------------------------------------ work open leads
    def work_leads(self):
        r = self.rng
        still = []
        for L in self.open_leads:
            retail = L["Channel"] == "Retail"
            created = L["_created"]
            age = (self.day_end - created) / DAY
            if L["_fr"] is None:
                if created >= self.day:
                    still.append(L)
                    continue
                if r.random() < (0.9 if age <= 2.5 else 0.03):
                    fr = self.day + pd.Timedelta(hours=9.5) + pd.Timedelta(minutes=float(r.lognormal(np.log(30), 0.8)))
                    L["_fr"] = fr
                    L.update(FirstResponseOn=ts(fr), LastContactedOn=ts(fr), Status="Contacted")
                    self.act("Lead", L["LeadID"], L["OwnerUserID"], L["OutletID"], L["Channel"], fr, "Call")
                still.append(L)
                continue
            window = 6 if retail else 21
            conv = gh.RETAIL_CONV.get(L["Source"], 0.3) if retail else gh.B2B_CONV.get(L["Source"], 0.4)
            if age <= window:
                if r.random() < 1 - (1 - conv) ** (1 / window):
                    lo = max(L["_fr"], self.day)
                    when = lo + (self.day_end - lo) * float(r.uniform(0.05, 0.95))
                    self.convert_lead(L, when)
                    continue
                if L["Status"] == "Contacted" and r.random() < 0.15:
                    L["Status"] = "Qualified"
                if r.random() < 0.45:
                    w = self.act("Lead", L["LeadID"], L["OwnerUserID"], L["OutletID"], L["Channel"],
                                 max(self.at(10, 19), L["_fr"]))
                    L["LastContactedOn"] = ts(w)
                still.append(L)
                continue
            lost_p, unresp_p = (0.08, 0.05) if retail else (0.05, 0.03)
            u = r.random()
            if u < lost_p + unresp_p:
                lost = u < lost_p
                w = self.act("Lead", L["LeadID"], L["OwnerUserID"], L["OutletID"], L["Channel"],
                             max(self.at(10, 19), L["_fr"]), "Call", None if lost else "No Answer")
                L.update(Status="Lost" if lost else "Unresponsive", LastContactedOn=ts(w))
                continue
            if r.random() < 0.03:
                w = self.act("Lead", L["LeadID"], L["OwnerUserID"], L["OutletID"], L["Channel"],
                             max(self.at(10, 19), L["_fr"]))
                L["LastContactedOn"] = ts(w)
            still.append(L)
        self.open_leads = still

    def convert_lead(self, L, when):
        r = self.rng
        retail = L["Channel"] == "Retail"
        owner = L["OwnerUserID"] or (self.retail_owner(L["OutletID"]) if retail else self.b2b_owner())
        acc = None
        if retail:
            cid = self.new_contact(dict(CustomerType="Individual", AccountID="", FullName=L["LeadName"], Designation="",
                                        IsDecisionMaker="True", Mobile=L["Mobile"], Email=L["Email"], City=L["City"],
                                        OwnerUserID=owner), when)
        else:
            if L["AccountID"]:
                acc = self.acc_by_id[L["AccountID"]]
            else:
                acc = self.new_account(L, owner, when)
                L["AccountID"] = acc["AccountID"]
            cands = self.contacts_by_acc.get(acc["AccountID"], [])
            dm = [c for c in cands if c["IsDecisionMaker"] == "True"]
            cid = self.choose(dm, cands)
            cid = cid["ContactID"] if cid else self.new_b2b_contact(acc, when)
        oid = self.create_opp(L, when, acc, cid, owner)
        L.update(Status="Converted", ConvertedOn=ts(when), LastContactedOn=ts(when), OpportunityID=oid,
                 OwnerUserID=owner)

    # ------------------------------------------------------------------ accounts & contacts
    def new_account(self, L, owner, when):
        r, h = self.rng, self.h
        name = L["CompanyName"]
        atype = L.get("_atype")
        if not atype:           # lead came from an earlier run: infer the type from the product of interest
            cands = [t for t, v in gh.ACCT_TYPES.items() if L["InterestedProductID"] in v["mix"]] or list(gh.ACCT_TYPES)
            atype = h.pick(cands, p=[gh.ACCT_TYPES[t]["share"] for t in cands])
        if L["LeadID"] in self.unlinked:
            atype = self.acc_by_id[self.unlinked[L["LeadID"]]]["AccountType"]
        entity = "G" if atype == "Government / PSU" else ("F" if "LLP" in name else ("C" if "Ltd" in name else "P"))
        industry = h.pick(gh.INDUSTRY_BY_TYPE[atype])
        city = L["City"] or h.pick(gh.KA_CITIES, p=gh.KA_CITY_W)
        slug = "".join(ch for ch in name.split(" Pvt")[0].split(" LLP")[0].lower() if ch.isalnum())[:22]
        pan = h.make_pan(entity, name)
        a = dict(AccountID=self.nid("account", "ACC", 5), AccountName=name, AccountType=atype, Industry=industry,
                 ParentAccountID="", City=city, State="Karnataka",
                 Website="" if atype == "Government / PSU" else slug + h.pick([".in", ".com", ".co.in"]),
                 GSTIN=h.make_gstin("29", pan), DUNS=h.new_duns() if r.random() < gh.ACCT_TYPES[atype]["duns"] else "",
                 ParentDUNS="", KN_Designation="Named" if r.random() < 0.10 else "",
                 EstimatedFleetSize=str(int(max(1, round(r.lognormal(np.log(gh.ACCT_TYPES[atype]["fleet"]), 0.7))))),
                 OwnerUserID=owner, CreatedOn=ts(when))
        h.used_names.add(name)
        rid = a["AccountID"]
        if L["LeadID"] in self.unlinked:
            orig = self.unlinked[L["LeadID"]]
            self.dup_ids.add(rid)
            self.orig_to_dup.setdefault(orig, rid)
            self.issue("duplicate_account", "dim_accounts", rid, orig,
                       "Same business as the related account; created again with a variant name")
        u = r.random()
        if u < 0.08:
            a["GSTIN"] = ""
            self.issue("missing_gstin", "dim_accounts", rid)
        elif u < 0.11:
            a["GSTIN"] = h.malform_gstin(a["GSTIN"])
            self.issue("malformed_gstin", "dim_accounts", rid, detail=a["GSTIN"])
        v = r.random()
        if v < 0.09 and industry in gh.INDUSTRY_VARIANTS:
            a["Industry"] = h.pick(gh.INDUSTRY_VARIANTS[industry])
            self.issue("nonstandard_industry", "dim_accounts", rid, detail=f"{a['Industry']} -> {industry}")
        elif v < 0.11:
            a["Industry"] = ""
            self.issue("missing_industry", "dim_accounts", rid)
        if city in gh.CITY_VARIANTS and r.random() < 0.07:
            a["City"] = h.pick(gh.CITY_VARIANTS[city])
            self.issue("nonstandard_city", "dim_accounts", rid, detail=f"'{a['City']}' -> {city}")
        self.accounts.append(a)
        self.acc_by_id[rid] = a
        for _ in range(int(r.integers(1, 3))):
            self.new_b2b_contact(a, when)
        return a

    def new_b2b_contact(self, a, when):
        r, h = self.rng, self.h
        nm = h.names(1)[0]
        first, last = nm.lower().replace("'", "").split(" ", 1)
        desig = h.pick(gh.B2B_DESIGNATIONS)
        email = (h.pick([f"{first}.{last}@{a['Website']}", f"{first}@{a['Website']}"]) if a["Website"] else
                 f"{first}{last}{r.integers(10, 99)}@{h.pick(gh.PERSONAL_EMAIL_DOMAINS)}").replace(" ", "")
        return self.new_contact(dict(CustomerType="Business", AccountID=a["AccountID"], FullName=nm, Designation=desig,
                                     IsDecisionMaker=str(desig in gh.DECISION_MAKERS), Mobile=h.mobiles(1)[0],
                                     Email=email, City=a["City"], OwnerUserID=a["OwnerUserID"]), when)

    def new_contact(self, c, when):
        r, h = self.rng, self.h
        c = dict(ContactID=self.nid("contact", "CON", 6), **c, CreatedOn=ts(when))
        u = r.random()
        if c["Email"] and u < 0.10:
            c["Email"] = ""
            self.issue("contact_missing_email", "dim_contacts", c["ContactID"])
        elif c["Email"] and u < 0.13:
            e = c["Email"]
            c["Email"] = h.pick([e.replace("@", ""), e.replace(".com", ".con").replace(".in", ".inn"),
                                 e.replace("@", " @"), e.split("@")[0] + "@"])
            self.issue("contact_invalid_email", "dim_contacts", c["ContactID"], detail=c["Email"])
        if r.random() < 0.03:
            m = c["Mobile"]
            c["Mobile"] = h.pick([m[:-1], m[0] * 10, "0" + m[:9], m[:5] + "-" + m[5:8]])
            self.issue("contact_invalid_mobile", "dim_contacts", c["ContactID"], detail=c["Mobile"])
        self.contacts.append(c)
        if c["AccountID"]:
            self.contacts_by_acc.setdefault(c["AccountID"], []).append(c)
        return c["ContactID"]

    # ------------------------------------------------------------------ opportunities
    def create_opp(self, L, when, acc, cid, owner):
        r, h = self.rng, self.h
        retail = L["Channel"] == "Retail"
        tender = L["Source"] == "Tender / RFP"
        t = self.tf(when)
        oid = self.nid("opportunity", "OPP", 6)
        if retail:
            prods = [L["InterestedProductID"] if r.random() < 0.8 else h.retail_products([t])[0]]
            qtys = [2 if r.random() < 0.03 else 1]
            disc_base = r.uniform(1.5, 5.5) + (2 if when.month in (10, 11) else 0) + (2 if when.month == 3 else 0)
        else:
            nl = r.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
            prods = [L["InterestedProductID"]] + [h.b2b_product(acc["AccountType"], t) for _ in range(nl - 1)]
            at = gh.ACCT_TYPES[acc["AccountType"]]
            qtys = [int(np.clip(round(r.lognormal(np.log(at["qty"]), 0.8)), 1, at["qmax"])) for _ in prods]
            disc_base = r.uniform(4, 9)
        units, value = 0, 0
        if r.random() < 0.01:
            self.issue("opportunity_missing_products", "fact_opportunities", oid, detail="No product lines, so revenue is 0")
        else:
            for p, q in zip(prods, qtys):
                unit = round(h.prod_price[p] * (1 + 0.03 * t / 365.25) * r.uniform(0.92, 1.20), -2)
                disc = disc_base + (0 if retail else (2 if q >= 25 else 0) + (3 if q >= 100 else 0))
                val = int(round(q * unit * (1 - disc / 100)))
                self.new_lines.append(dict(LineID=self.nid("line", "OPL", 7), OpportunityID=oid, ProductID=p,
                                           Quantity=str(q), UnitPriceINR=str(int(unit)), DiscountPct=str(round(disc, 2)),
                                           LineValueINR=str(val)))
                units += q
                value += val
            self.first_prod[oid] = prods[0]
        label = h.prod_label[prods[0]]
        name = f"{label} - {L['LeadName']}" if retail else f"{acc['AccountName'][:40]} - {units or '?'}x {label}"
        exp_close = when + pd.Timedelta(days=float((25 if retail else (120 if tender else 75)) * r.uniform(0.7, 1.4)))
        fin = h.pick(["Cash", "Bank Loan", "OEM Partner Finance"], p=[0.25, 0.6, 0.15]) if retail else \
            h.pick(["Corporate Purchase", "Leasing", "Bank Loan"], p=[0.4, 0.25, 0.35])
        o = dict(OpportunityID=oid, OpportunityName=name, LeadID=L["LeadID"], Channel=L["Channel"],
                 OutletID=L["OutletID"], AccountID=acc["AccountID"] if acc else "", ContactID=cid, OwnerUserID=owner,
                 LeadSource=L["Source"], CreatedOn=ts(when), StageNo="1", Stage=gh.STAGES[0], Status="Open",
                 Probability=str(gh.STAGE_PROB[0]), TotalUnits=str(units), EstimatedRevenueINR=str(value),
                 ExpectedCloseDate=exp_close.strftime("%Y-%m-%d"), ActualCloseDate="", LastActivityOn=ts(when),
                 FinanceType=fin, HasExchangeVehicle=str(bool(retail and r.random() < 0.3)), IsTender=str(bool(tender)),
                 LostReason="")
        self.opps.append(o)
        self.prep_opp(o)
        self.open_opps.append(o)
        self.add_hist(oid, 0, when)
        self.act("Opportunity", oid, owner, o["OutletID"], o["Channel"], when,
                 "Showroom Visit" if retail else "Meeting", "Completed")
        return oid

    def add_hist(self, oid, stage_idx, when):
        row = dict(StageHistoryID=f"STH-{self.next_hist:07d}", OpportunityID=oid, StageNo=str(stage_idx + 1),
                   Stage=gh.STAGES[stage_idx], EnteredOn=ts(when), ExitedOn="", DaysInStage="", _entered=when)
        self.next_hist += 1
        self.hist.append(row)
        if stage_idx <= 4:
            self.open_hist[oid] = row

    def close_hist(self, oid, when):
        row = self.open_hist.pop(oid)
        row["ExitedOn"] = ts(when)
        row["DaysInStage"] = str(round((when - row["_entered"]) / DAY, 2))

    def destined_cold(self, o):
        n = int(o["OpportunityID"][4:])
        return (n * 2654435761) % 1000 < 35 and (self.day - o["_created"]).days > 5 + (n * 97) % 40

    def work_opportunities(self):
        r = self.rng
        still = []
        for o in self.open_opps:
            oid, retail = o["OpportunityID"], o["_retail"]
            idle = (self.day - o["_last"]) / DAY
            if idle > 30 or self.destined_cold(o):
                if idle > 45 and r.random() < 0.003:            # occasionally a manager closes a dead deal
                    self.finish(o, self.at(10, 19), won=False, reason="CRM Cleanup - No Response")
                    continue
                still.append(o)
                continue
            hrow = self.open_hist.get(oid)
            s = int(o["StageNo"]) - 1
            dwell = (gh.RETAIL_DWELL if retail else gh.B2B_DWELL)[s]
            dwell = dwell * (1.8 if (o["_tender"] and not retail) else 1.0) + (10 if (s == 4 and o["_ev"]) else 0)
            if hrow is not None and hrow["_entered"] < self.day and r.random() < min(0.9, 1 / dwell):
                when = max(self.at(10, 19), hrow["_entered"] + pd.Timedelta(minutes=5))
                adv = (gh.RETAIL_ADV if retail else gh.B2B_ADV)[s]
                if r.random() < adv:
                    self.close_hist(oid, when)
                    if s == 4:
                        self.finish(o, when, won=True, hist_closed=True)
                        continue
                    self.add_hist(oid, s + 1, when)
                    o.update(StageNo=str(s + 2), Stage=gh.STAGES[s + 1], Probability=str(gh.STAGE_PROB[s + 1]))
                    w = self.act("Opportunity", oid, o["OwnerUserID"], o["OutletID"], o["Channel"], when,
                                 STAGE_ACT[s + 1][0 if retail else 1], "Completed")
                    o["_last"], o["LastActivityOn"] = w, ts(w)
                else:
                    self.finish(o, when, won=False)
                    continue
            for _ in range(r.poisson(1.0 if retail else 0.5)):
                w = self.act("Opportunity", oid, o["OwnerUserID"], o["OutletID"], o["Channel"],
                             max(self.at(9, 20), o["_created"]))
                if w > o["_last"]:
                    o["_last"], o["LastActivityOn"] = w, ts(w)
            still.append(o)
        self.open_opps = still

    def finish(self, o, when, won, reason=None, hist_closed=False):
        oid, retail = o["OpportunityID"], o["_retail"]
        if not hist_closed and oid in self.open_hist:
            self.close_hist(oid, when)
        self.add_hist(oid, 5 if won else 6, when)
        if not won and reason is None:
            reason = self.h.pick(gh.RETAIL_LOST, p=gh.RETAIL_LOST_W) if retail else self.h.pick(gh.B2B_LOST, p=gh.B2B_LOST_W)
        o.update(StageNo="6" if won else "7", Stage=gh.STAGES[5 if won else 6], Status="Won" if won else "Lost",
                 Probability="100" if won else "0", ActualCloseDate=when.strftime("%Y-%m-%d"),
                 LostReason="" if won else reason)
        w = self.act("Opportunity", oid, o["OwnerUserID"], o["OutletID"], o["Channel"], when,
                     ("Vehicle Delivery" if retail else "Meeting") if won else "Call", "Completed" if won else None)
        o["_last"], o["LastActivityOn"] = w, ts(w)

    # ------------------------------------------------------------------ reporting requests (SLA log)
    def work_requests(self):
        r = self.rng
        for q in self.requests:
            if q["Status"] == "In Progress" and r.random() < 0.6:
                when = max(self.at(9.5, 19), pd.Timestamp(q["RequestedOn"]) + pd.Timedelta(minutes=10))
                q.update(Status="Delivered", DeliveredOn=ts(min(when, self.day_end - pd.Timedelta(seconds=1))),
                         ReworkRequired=str(bool(r.random() < 0.08)))
        weekday = self.day.dayofweek < 5
        requesters = [u for role in ("GM - Sales", "Outlet Sales Manager", "Head - Corporate & Fleet Sales",
                                     "Key Account Manager") for u in self.pool(role)]
        cats = list(REQ_CATS)
        cw = np.array([REQ_CATS[c][2] for c in cats], float)
        for _ in range(r.poisson(1.9 * self.scale * (7 / 5 if weekday else 0.15))):
            cat = cats[r.choice(len(cats), p=cw / cw.sum())]
            rtype, pri, _ = REQ_CATS[cat]
            if rtype == "Ad hoc" and r.random() < 0.2:
                pri = "P1"
            req = self.at(9.5, 18.5)
            due = req + pd.Timedelta(hours=SLA_H[pri])
            done = req + pd.Timedelta(hours=float(SLA_H[pri] * r.lognormal(np.log(0.55), 0.6)))
            status, delivered = "Delivered", ts(done)
            if r.random() < 0.03:
                status, delivered = "Cancelled", ""
            elif done >= self.day_end:
                status, delivered = "In Progress", ""
            self.requests.append(dict(
                RequestID=self.nid("request", "REQ", 5), RequestedOn=ts(req),
                RequestedBy="OEM Regional Office" if cat == "OEM Submission" else self.choose(requesters),
                Category=cat, RequestType=rtype, Priority=pri, SLAHours=str(SLA_H[pri]), DueOn=ts(due),
                DeliveredOn=delivered, Status=status,
                AssignedToUserID=self.choose(self.pool("Sales Ops / MIS Analyst"), self.pool("GM - Sales")),
                ReworkRequired=str(bool(status == "Delivered" and r.random() < 0.08))))

    # ------------------------------------------------------------------ activities & snapshots
    def flush_activities(self):
        if not self.acts:
            return
        df = pd.DataFrame(self.acts).sort_values("ActivityOn", kind="stable")
        df.insert(0, "ActivityID", [self.nid("activity", "ACT", 8) for _ in range(len(df))])
        fy = self.day.year if self.day.month >= 4 else self.day.year - 1
        df["ActivityOn"] = df["ActivityOn"].dt.strftime(FMT)
        path = self.p("activities", f"fact_activities_FY{fy}-{str(fy + 1)[-2:]}.csv")
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    def snapshot_day(self):
        agg = {}
        ds = self.day.strftime("%Y-%m-%d")
        for o in self.open_opps:
            k = (o["OutletID"], o["Channel"], int(o["StageNo"]))
            v = agg.setdefault(k, [0, 0, 0, 0, 0])
            v[0] += 1
            v[1] += int(o["TotalUnits"] or 0)
            v[2] += int(o["EstimatedRevenueINR"] or 0)
            v[3] += (self.day_end - o["_last"]) / DAY > 30
            v[4] += o["ExpectedCloseDate"] < ds
        for (outlet, ch, s), v in sorted(agg.items()):
            self.snap_rows.append([ds, outlet, ch, s, gh.STAGES[s - 1]] + [int(x) for x in v])

    def backfill_snapshot(self):
        """First run only: rebuild the daily pipeline snapshot for the whole history from stage history."""
        print("  backfilling fact_pipeline_snapshot from stage history (first run only)...")
        h = pd.DataFrame([x for x in self.hist if int(x["StageNo"]) <= 5])[["OpportunityID", "StageNo", "EnteredOn", "ExitedOn"]]
        ent = pd.to_datetime(h["EnteredOn"])
        ex = pd.to_datetime(h["ExitedOn"], errors="coerce")
        first = ent.dt.floor("D")
        last = (ex.dt.floor("D") - DAY).fillna(self.as_of)
        n = ((last - first) / DAY).astype(int) + 1
        keep = n > 0
        h, first, n = h[keep].reset_index(drop=True), first[keep].reset_index(drop=True), n[keep].values
        rep = np.repeat(np.arange(len(h)), n)
        offs = np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n)
        x = pd.DataFrame({"OpportunityID": h["OpportunityID"].values[rep], "StageNo": h["StageNo"].values[rep].astype(int),
                          "Day": first.values[rep] + offs * np.timedelta64(1, "D")})
        o = pd.DataFrame(self.opps)[["OpportunityID", "OutletID", "Channel", "TotalUnits", "EstimatedRevenueINR",
                                     "ExpectedCloseDate", "CreatedOn"]]
        x = x.merge(o, on="OpportunityID", how="left")
        x["Snap"] = x["Day"] + DAY
        acts = []
        for f in sorted(os.listdir(self.p("activities"))):
            a = pd.read_csv(self.p("activities", f), usecols=["ActivityOn", "RegardingType", "RegardingID"], dtype=str)
            acts.append(a[a["RegardingType"] == "Opportunity"])
        a = pd.concat(acts)
        a = pd.DataFrame({"OpportunityID": a["RegardingID"].values, "LastAct": pd.to_datetime(a["ActivityOn"]).values})
        x = pd.merge_asof(x.sort_values("Snap"), a.sort_values("LastAct"), left_on="Snap", right_on="LastAct",
                          by="OpportunityID", direction="backward")
        x["LastAct"] = x["LastAct"].fillna(pd.to_datetime(x["CreatedOn"]))
        x["Stale"] = (x["Snap"] - x["LastAct"]) / DAY > 30
        x["Overdue"] = x["ExpectedCloseDate"] < x["Day"].dt.strftime("%Y-%m-%d")
        x["Units"] = pd.to_numeric(x["TotalUnits"]).fillna(0)
        x["Value"] = pd.to_numeric(x["EstimatedRevenueINR"]).fillna(0)
        g = x.groupby(["Day", "OutletID", "Channel", "StageNo"]).agg(
            OpenDeals=("OpportunityID", "size"), OpenUnits=("Units", "sum"), OpenValueINR=("Value", "sum"),
            StaleDeals=("Stale", "sum"), OverdueDeals=("Overdue", "sum")).reset_index()
        g.insert(0, "SnapshotDate", g.pop("Day").dt.strftime("%Y-%m-%d"))
        g.insert(4, "Stage", [gh.STAGES[s - 1] for s in g["StageNo"]])
        for c in ["OpenUnits", "OpenValueINR", "StaleDeals", "OverdueDeals"]:
            g[c] = g[c].astype(int)
        g[SNAP_COLS].to_csv(self.p("fact_pipeline_snapshot.csv"), index=False)

    # ------------------------------------------------------------------ answer key (rule-based part)
    def rule_based_issues(self):
        asof = self.as_of.strftime("%Y-%m-%d")
        end = self.as_of + DAY
        inactive = {u["UserID"] for u in self.users if u["ExitDate"]}
        out = []

        def add(t, tbl, rid, rel="", det=""):
            out.append(dict(IssueType=t, Table=tbl, RecordID=rid, RelatedRecordID=rel, Detail=det))
        for a in self.accounts:
            if a["KN_Designation"] == "Key" and not a["DUNS"]:
                add("key_account_missing_duns", "dim_accounts", a["AccountID"])
            if a["OwnerUserID"] in inactive:
                add("account_owner_inactive", "dim_accounts", a["AccountID"], a["OwnerUserID"])
        for o in self.opps:
            if o["Status"] != "Open":
                continue
            if o["OwnerUserID"] in inactive:
                add("open_opportunity_owner_inactive", "fact_opportunities", o["OpportunityID"], o["OwnerUserID"])
            idle = (end - pd.Timestamp(o["LastActivityOn"])) / DAY
            if idle > 30:
                add("stale_open_opportunity", "fact_opportunities", o["OpportunityID"],
                    det=f"{int(idle)} days since last activity")
            if idle > 60:
                add("abandoned_open_opportunity", "fact_opportunities", o["OpportunityID"],
                    det="No activity for 60+ days but never closed")
            if o["ExpectedCloseDate"] < asof:
                add("open_opportunity_past_expected_close", "fact_opportunities", o["OpportunityID"])
        cutoff = ts(end - 30 * DAY)
        for L in self.leads:
            if L["Status"] in OPEN_LEAD and L["CreatedOn"] < cutoff:
                add("unconverted_lead_over_30_days", "fact_leads", L["LeadID"], det=L["Status"])
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ save
    def save(self):
        def write(rows, name):
            cols = self.cols[name]
            pd.DataFrame(rows).reindex(columns=cols).fillna("").to_csv(self.p(name), index=False)

        asof = self.as_of.strftime("%Y-%m-%d")
        for u in self.users:
            u["IsActive"] = str(u["ExitDate"] == "" and u["JoinDate"] <= asof)
        write(self.users, "dim_users.csv")
        write(self.accounts, "dim_accounts.csv")
        write(self.contacts, "dim_contacts.csv")
        write(self.leads, "fact_leads.csv")
        write(self.opps, "fact_opportunities.csv")
        write(self.hist, "fact_stage_history.csv")
        write(self.requests, "fact_report_requests.csv")
        if self.new_lines:
            pd.DataFrame(self.new_lines)[self.cols["fact_opportunity_lines.csv"]].to_csv(
                self.p("fact_opportunity_lines.csv"), mode="a", header=False, index=False)
        self.targets.to_csv(self.p("fact_targets.csv"), index=False)
        if self.snap_rows:
            pd.DataFrame(self.snap_rows, columns=SNAP_COLS).to_csv(self.p("fact_pipeline_snapshot.csv"), mode="a",
                                                                   header=False, index=False)
        key = self.key[~self.key["IssueType"].isin(RULE_BASED)]
        key = pd.concat([key, pd.DataFrame(self.issues), self.rule_based_issues()], ignore_index=True).fillna("")
        key.to_csv(self.p("_data_quality_answer_key.csv"), index=False)

        manifest = []
        for root, _, files in os.walk(self.data):
            for f in sorted(files):
                if not f.endswith(".csv") or f.startswith("_manifest") or f.startswith("_validation"):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, self.data).replace(os.sep, "/")
                with open(full, "rb") as fh:
                    rows = sum(1 for _ in fh) - 1
                table = "fact_activities" if rel.startswith("activities/") else f[:-4]
                manifest.append(dict(Path=rel, Table=table, Rows=rows, SizeMB=round(os.path.getsize(full) / 1e6, 2),
                                     DataAsOf=asof))
        m = pd.DataFrame(manifest).sort_values("Path")
        m.to_csv(self.p("_manifest.csv"), index=False)

        s = self.summary
        s["previous_row_counts"] = self.prev_counts
        s["row_counts"] = dict(zip(m["Path"], m["Rows"].astype(int)))
        s["data_as_of"] = asof
        s["last_update_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
        s["next_ids"] = self.nx
        s["planted_issue_counts"] = key["IssueType"].value_counts().to_dict()
        with open(self.p("_generation_summary.json"), "w") as f:
            json.dump(s, f, indent=2, default=int)

    # ------------------------------------------------------------------ main loop
    def check_files_closed(self):
        """Stop before changing anything if a data file is open in Excel / Power BI (Windows locks it)."""
        locked = []
        for root, _, files in os.walk(self.data):
            for f in files:
                if f.endswith((".csv", ".json")):
                    path = os.path.join(root, f)
                    try:
                        with open(path, "a"):
                            pass
                    except PermissionError:
                        locked.append(os.path.relpath(path, self.data))
        if locked:
            print("\nSTOPPED - nothing was changed.")
            print("These files are open in another program (usually Excel or Power BI Desktop):")
            for f in locked:
                print(f"   {f}")
            print("Close them, then run this script again.")
            raise SystemExit(1)

    def run(self, until):
        t0 = time.time()
        self.check_files_closed()
        if not os.path.exists(self.p("fact_pipeline_snapshot.csv")):
            self.backfill_snapshot()
        day, n_days = self.as_of + DAY, 0
        counts0 = (len(self.leads), len(self.opps), len(self.accounts))
        while day <= until:
            self.day, self.day_end = day, day + DAY
            self.rng = np.random.default_rng([self.seed, day.toordinal()])
            self.h.rng, self.h.T = self.rng, (day - self.START).days + 1
            self.acts = []
            self.build_pools()
            self.attrition()
            self.build_pools()
            self.monthly_targets()
            self.new_retail_leads()
            self.new_b2b_leads()
            self.work_leads()
            self.work_opportunities()
            self.work_requests()
            self.flush_activities()
            self.snapshot_day()
            self.as_of = day
            n_days += 1
            print(f"  {day.date()}  leads={len(self.leads):,}  open deals={len(self.open_opps):,}  "
                  f"activities today={len(self.acts):,}")
            day += DAY
        self.save()
        added = (len(self.leads) - counts0[0], len(self.opps) - counts0[1], len(self.accounts) - counts0[2])
        print(f"Done: {n_days} day(s) added, data now as of {self.as_of.date()} "
              f"(+{added[0]:,} leads, +{added[1]:,} opportunities, +{added[2]:,} accounts) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Move the synthetic CRM forward to yesterday")
    ap.add_argument("--data", default="data", help="data folder created by generate_history.py")
    ap.add_argument("--until", default=None, help="last day to simulate, YYYY-MM-DD (default: yesterday)")
    args = ap.parse_args()
    until = pd.Timestamp(args.until) if args.until else pd.Timestamp.today().normalize() - DAY
    u = Updater(args.data)
    if until <= u.as_of:
        print(f"Data is already up to date (as of {u.as_of.date()}). Refreshing manifest only.")
    u.run(until)