# Anvaya Motors: Dealership Sales Ops Dataset (Data Dictionary)

**Anvaya Motors** is a fictional Tata Motors passenger-vehicle dealer group in Karnataka. It has 7 retail showrooms and a Corporate & Fleet desk that sells to cab operators, employee-transport companies, corporates, leasing firms, government offices, hospitals and hotels.

All data is synthetic. The people, companies, phone numbers, GSTIN and DUNS values, and all numbers are randomly generated. Model names are used only as product labels. Prices and launch dates are indicative approximations, not an official price list.

## How to generate

```
pip install pandas numpy
python generate_history.py              # 3 years ending yesterday, ~2M rows, ~210 MB, ~20 seconds
python generate_history.py --scale 2    # about double the volume
python generate_history.py --seed 7     # a different but reproducible dataset
```

The same seed and as-of date always produce identical data.

## Default volumes

| File | Rows | Grain |
|---|---|---|
| dim_outlets | 9 | One per showroom / desk / head office |
| dim_users | ~216 | One per employee who worked there in the period (includes people who left) |
| dim_products | 21 | One per model + powertrain |
| dim_accounts | ~4,950 | One per B2B customer record, **including duplicates** |
| dim_contacts | ~48,500 | Business contacts + individual retail buyers |
| fact_leads | ~106,600 | One per enquiry |
| fact_opportunities | ~43,000 | One per deal |
| fact_opportunity_lines | ~43,700 | One per product on a deal |
| fact_stage_history | ~177,000 | One per stage an opportunity entered |
| activities/fact_activities_FY*.csv | ~1.57M | One per call, WhatsApp, visit, test drive, meeting... Split by Indian financial year (Apr–Mar) |
| fact_targets | ~288 | Outlet × month |
| fact_report_requests | ~1,500 | One per reporting request to the MIS team |
| fact_pipeline_snapshot | ~40,000, +~30/day | Open pipeline per day × outlet × stage (created by daily_update.py) |
| _data_quality_answer_key | ~26,000 | Every planted or rule-detectable data-quality issue |
| _manifest.csv | 1 per file | List of every data file with row counts; Power BI uses it to find the activity files |
| _validation_log.csv | ~44 per run | Result of every validation check on every run (for a data-health page) |

## Relationships (star schema)

```
dim_outlets  1──*  dim_users, fact_leads, fact_opportunities, fact_targets, activities
dim_users    1──*  dim_accounts (Owner), fact_leads, fact_opportunities, activities, report_requests (AssignedTo)
dim_accounts 1──*  dim_contacts, fact_leads, fact_opportunities
dim_accounts 1──*  dim_accounts (ParentAccountID → AccountID, self-hierarchy)
dim_contacts 1──*  fact_opportunities
dim_products 1──*  fact_opportunity_lines, fact_leads (InterestedProductID)
fact_opportunities 1──* fact_opportunity_lines, fact_stage_history
activities: RegardingID points to a LeadID or OpportunityID (see RegardingType)
```

Build your own Date table in DAX. The data follows the Indian financial year (April start).

## Key columns

**dim_users**: `Role`, `ManagerUserID` (hierarchy), `JoinDate`, `ExitDate`, `IsActive`, `MonthlyUnitTarget`. Attrition is realistic (high for sales consultants), which is why records owned by people who have left show up as a hygiene issue.

**dim_accounts**: `AccountType`, `Industry`, `ParentAccountID`, `City`, `State`, `Website`, `GSTIN`, `DUNS`, `ParentDUNS` (the D&B family tree), `KN_Designation` (Key / Named / blank), `EstimatedFleetSize` (use for whitespace analysis), `OwnerUserID`, `CreatedOn`.

**fact_leads**: `Channel`, `Source`, `Status` (New / Contacted / Qualified / Converted / Lost / Unresponsive), `FirstResponseOn` (for response-time SLA), `LastContactedOn`, `ConvertedOn`, `OpportunityID`, `IsExistingCustomer`, `InterestedProductID`.

**fact_opportunities**: `Stage` (1-Enquiry → 2-Test Drive / Demo → 3-Quotation → 4-Negotiation → 5-Booking / PO → 6-Closed Won / 7-Closed Lost), `Status` (Open / Won / Lost), `Probability`, `TotalUnits`, `EstimatedRevenueINR`, `ExpectedCloseDate`, `ActualCloseDate`, `LastActivityOn`, `LeadSource`, `FinanceType`, `HasExchangeVehicle`, `IsTender`, `LostReason`.

**fact_stage_history**: `EnteredOn`, `ExitedOn`, `DaysInStage`. Use it for stage velocity, conversion funnels, and "pipeline as of any date" analysis.

**fact_pipeline_snapshot**: `SnapshotDate`, `OutletID`, `Channel`, `StageNo`, `Stage`, `OpenDeals`, `OpenUnits`, `OpenValueINR`, `StaleDeals` (no activity for 30+ days), `OverdueDeals` (expected close date already passed). The first run backfills it for the whole history from stage history; after that one day is appended per run. Treat the first two months as warm-up, because the history starts with an empty pipeline.

**fact_report_requests**: `Category`, `RequestType` (Standing / Ad hoc), `Priority` (P1 = 4h, P2 = 24h, P3 = 72h), `SLAHours`, `RequestedOn`, `DueOn`, `DeliveredOn`, `Status`, `ReworkRequired`.

## Realistic patterns built in

- Seasonality: peaks in the festive season (Oct–Nov) and at financial year-end (March), a dip during the monsoon (Jun–Jul), and more walk-ins on weekends.
- About 7% year-on-year growth, plus a rising EV share (fleet much higher than retail).
- New launches appear only after their launch date (Punch EV, Curvv, Curvv EV, Nexon CNG, Harrier EV).
- Online leads that arrive at night wait until the next morning for a first response.
- Discounts rise during the festive season, at year-end, and on large fleet orders.
- B2B deals are slower. Government tenders take longest.

## Planted data-quality issues (the CRM hygiene part of the JD)

Every issue below is listed in `_data_quality_answer_key.csv` with its record ID. Build your checks in Power Query / DAX first, **then** compare your results against the answer key.

| IssueType | What to detect |
|---|---|
| duplicate_account | Same business entered twice with a variant name (UPPERCASE, "Private Limited", "M/s", typos, "&" vs "and"). Matches via domain / GSTIN / fuzzy name. Some later deals are logged on the duplicate, which splits the customer's history |
| missing_gstin / malformed_gstin | Blank GSTIN, or wrong length, lowercase, embedded space, or wrong check digit |
| nonstandard_industry / missing_industry | "IT", "BFSI", "Pharma" etc. instead of standard values |
| nonstandard_city | Bangalore / BLR / Mysore / Hubli / Belgaum instead of the official names |
| missing_parent_link | Child account with a blank ParentAccountID (the name and ParentDUNS still reveal the group) |
| orphan_parent_reference | ParentAccountID points to an account that doesn't exist |
| parent_misaligned_with_dnb | CRM parent disagrees with the D&B family tree (ParentDUNS) |
| kn_designation_inconsistent | Child's Key / Named tag differs from its parent's |
| key_account_missing_duns | Key accounts should always carry a DUNS number |
| account_owner_inactive / open_opportunity_owner_inactive | Records still owned by employees who left |
| duplicate_contact | Same person twice ("Ramesh Kumar" vs "RAMESH KUMAR" / "Mr. Ramesh Kumar"), same mobile |
| contact_missing_email / contact_invalid_email / contact_invalid_mobile | Bad or missing contact data |
| duplicate_lead | Same retail customer enquired again through another source |
| lead_unassigned | Lead with no owner |
| b2b_lead_not_linked_to_existing_account | Lead for a company that already exists as an account, but isn't linked to it |
| unconverted_lead_over_30_days | Leads still New / Contacted / Qualified after 30 days |
| stale_open_opportunity / abandoned_open_opportunity | Open deals with no activity for 30+ days (stale) or 60+ days (abandoned) |
| open_opportunity_past_expected_close | Open deals whose expected close date has already passed |
| opportunity_missing_products | Deal with no product lines, so revenue is 0 |

**A trap worth knowing about:** branches of the same legal entity in the same state legitimately share one GSTIN. Deduplicating by GSTIN alone would wrongly merge real branches. Check `ParentAccountID` and the "- City Branch" names before treating a shared GSTIN as a duplicate. Explaining this in an interview shows you understand the data, not just the tool.

## Daily updates (Phase 2)

`daily_update.py` moves the CRM forward to yesterday and plants the same kinds of problems in new records. A few things only appear after daily updates start: the `Vehicle Delivery` activity type on won retail deals, the lost reason `CRM Cleanup - No Response` when a manager closes a long-dead deal, and new staff replacing people who leave. Rule-based issue types (inactive owners, stale deals, overdue deals, old unconverted leads, Key accounts missing DUNS) are recalculated on every run, so the answer key always matches the current data.

## Loading into Power BI

- Load `activities/` with **Get Data → Folder** and combine the files, so new financial-year files are picked up automatically.
- Set data types explicitly. Keep IDs, GSTIN, DUNS and Mobile as **Text**, or leading characters and formats will be lost.
- Timestamps are ISO format (`YYYY-MM-DD HH:MM:SS`) and parse correctly under any locale.
- `_generation_summary.json` stores the as-of date and the next free ID numbers. Phase 2 (the daily update job) will read it.
