#!/usr/bin/env python3
"""
generate_history.py  -  Phase 1 of the "Dealership Sales Ops Command Center" portfolio project

Generates ~3 years of synthetic CRM history for a FICTIONAL Tata Motors passenger-vehicle
dealership group ("Anvaya Motors", Karnataka) with two sales channels:
  * Retail              - 7 showrooms selling to individual buyers
  * Corporate & Fleet   - a B2B desk selling to cab operators, employee-transport firms,
                          corporates, leasing companies, government, hospitals, hotels

Everything is synthetic: people, companies, phone numbers, GSTIN / DUNS values and all numbers
are randomly generated. Tata model names are used only as product labels, and prices / launch
dates are indicative approximations, NOT an official price list.

The data deliberately contains CRM hygiene problems (duplicates, missing / malformed GSTIN,
broken account hierarchies, stale deals, bad contact data ...). Every planted problem is
listed in _data_quality_answer_key.csv so you can check that your Power Query / DAX logic
catches them.

Usage
    pip install pandas numpy
    python generate_history.py                    # defaults: 3 years ending yesterday, ~2M rows
    python generate_history.py --scale 2          # ~2x the volume
    python generate_history.py --as-of 2026-09-22 --seed 7 --out data
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------
# Static configuration
# --------------------------------------------------------------------------------------------
GROUP_NAME = "Anvaya Motors"               # fictional dealer group
STAFF_EMAIL_DOMAIN = "anvayamotors.example"  # .example is a reserved, never-real domain

# id, outlet name, city, zone, channel, retail volume weight
OUTLETS = [
    ("HQ",    "Group Head Office",       "Bengaluru", "Bengaluru Central", "Head Office",       0.0),
    ("OUT01", "Whitefield",              "Bengaluru", "Bengaluru East",    "Retail",            1.25),
    ("OUT02", "Hebbal",                  "Bengaluru", "Bengaluru North",   "Retail",            1.15),
    ("OUT03", "Electronic City",         "Bengaluru", "Bengaluru South",   "Retail",            1.10),
    ("OUT04", "Jayanagar",               "Bengaluru", "Bengaluru South",   "Retail",            1.00),
    ("OUT05", "Mysuru",                  "Mysuru",    "South Karnataka",   "Retail",            0.80),
    ("OUT06", "Hubballi",                "Hubballi",  "North Karnataka",   "Retail",            0.70),
    ("OUT07", "Mangaluru",               "Mangaluru", "Coastal Karnataka", "Retail",            0.75),
    ("OUT08", "Corporate & Fleet Desk",  "Bengaluru", "Bengaluru Central", "Corporate & Fleet", 0.0),
]
RETAIL_OUTLETS = [o[0] for o in OUTLETS if o[4] == "Retail"]

# id, model, powertrain, segment, fuel, indicative price (lakh INR), launch, fleet-only, retail popularity
PRODUCTS = [
    ("PRD01", "Tiago",      "Petrol", "Hatchback",         "Petrol",   5.6,  "2020-01-22", 0, 10),
    ("PRD02", "Tiago",      "CNG",    "Hatchback",         "CNG",      6.6,  "2022-01-19", 0, 6),
    ("PRD03", "Tiago EV",   "EV",     "Hatchback",         "Electric", 9.0,  "2022-09-28", 0, 2.5),
    ("PRD04", "Tigor",      "Petrol", "Compact Sedan",     "Petrol",   6.6,  "2020-01-22", 0, 3),
    ("PRD05", "Tigor",      "CNG",    "Compact Sedan",     "CNG",      7.7,  "2022-01-19", 0, 3),
    ("PRD06", "Tigor EV",   "EV",     "Compact Sedan",     "Electric", 12.5, "2021-08-31", 0, 0.8),
    ("PRD07", "Xpres-T EV", "EV",     "Compact Sedan",     "Electric", 12.5, "2021-07-01", 1, 0),
    ("PRD08", "Punch",      "Petrol", "Micro SUV",         "Petrol",   6.4,  "2021-10-18", 0, 12),
    ("PRD09", "Punch",      "CNG",    "Micro SUV",         "CNG",      7.3,  "2023-08-04", 0, 7),
    ("PRD10", "Punch EV",   "EV",     "Micro SUV",         "Electric", 11.0, "2024-01-17", 0, 3),
    ("PRD11", "Altroz",     "Petrol", "Premium Hatchback", "Petrol",   7.0,  "2020-01-22", 0, 6),
    ("PRD12", "Altroz",     "CNG",    "Premium Hatchback", "CNG",      7.8,  "2023-05-22", 0, 3),
    ("PRD13", "Nexon",      "Petrol", "Compact SUV",       "Petrol",   8.9,  "2017-09-21", 0, 11),
    ("PRD14", "Nexon",      "Diesel", "Compact SUV",       "Diesel",   10.8, "2017-09-21", 0, 5),
    ("PRD15", "Nexon",      "CNG",    "Compact SUV",       "CNG",      9.2,  "2024-09-24", 0, 4),
    ("PRD16", "Nexon EV",   "EV",     "Compact SUV",       "Electric", 13.5, "2020-01-28", 0, 3.5),
    ("PRD17", "Curvv",      "Petrol", "Coupe SUV",         "Petrol",   11.0, "2024-09-02", 0, 4),
    ("PRD18", "Curvv EV",   "EV",     "Coupe SUV",         "Electric", 17.5, "2024-08-07", 0, 1.2),
    ("PRD19", "Harrier",    "Diesel", "Mid-size SUV",      "Diesel",   16.0, "2019-01-23", 0, 4),
    ("PRD20", "Harrier EV", "EV",     "Mid-size SUV",      "Electric", 22.0, "2025-06-03", 0, 0.8),
    ("PRD21", "Safari",     "Diesel", "Mid-size SUV",      "Diesel",   16.5, "2021-02-22", 0, 3.5),
]

# B2B account types: share of accounts, fleet-size median, qty-per-line median / max,
# repeat enquiries per year, DUNS probability, product mix (product id -> weight)
ACCT_TYPES = {
    "Employee Transport Services": dict(share=0.14, fleet=120, qty=6, qmax=120, repeat=0.45, duns=0.45,
        mix={"PRD05": 12, "PRD07": 10, "PRD06": 6, "PRD02": 6, "PRD09": 3, "PRD15": 2, "PRD03": 3}),
    "Cab Fleet Operator": dict(share=0.22, fleet=25, qty=4, qmax=60, repeat=0.5, duns=0.08,
        mix={"PRD05": 12, "PRD07": 10, "PRD06": 6, "PRD02": 6, "PRD09": 3, "PRD15": 2, "PRD03": 3}),
    "Car Rental & Self-Drive": dict(share=0.08, fleet=40, qty=3, qmax=40, repeat=0.3, duns=0.25,
        mix={"PRD08": 6, "PRD13": 5, "PRD01": 4, "PRD09": 3, "PRD11": 3, "PRD16": 2}),
    "Corporate (Staff & Executive Cars)": dict(share=0.30, fleet=15, qty=2, qmax=25, repeat=0.18, duns=0.65,
        mix={"PRD13": 6, "PRD16": 6, "PRD19": 4, "PRD21": 4, "PRD17": 3, "PRD18": 2, "PRD10": 2, "PRD20": 1}),
    "Leasing & Subscription": dict(share=0.05, fleet=400, qty=8, qmax=150, repeat=0.5, duns=0.70,
        mix={"PRD16": 6, "PRD10": 5, "PRD18": 3, "PRD13": 4, "PRD19": 2, "PRD03": 3}),
    "Government / PSU": dict(share=0.05, fleet=30, qty=4, qmax=50, repeat=0.15, duns=0.05,
        mix={"PRD13": 5, "PRD04": 3, "PRD21": 3, "PRD16": 3, "PRD06": 2, "PRD19": 2}),
    "Hospitals & Education": dict(share=0.09, fleet=8, qty=2, qmax=12, repeat=0.15, duns=0.30,
        mix={"PRD13": 3, "PRD21": 2, "PRD01": 2, "PRD08": 3, "PRD05": 2}),
    "Hotels & Travel": dict(share=0.07, fleet=10, qty=2, qmax=10, repeat=0.18, duns=0.35,
        mix={"PRD21": 4, "PRD19": 3, "PRD13": 3, "PRD17": 2}),
}

INDUSTRY_BY_TYPE = {
    "Employee Transport Services": ["Transportation Services"],
    "Cab Fleet Operator": ["Transportation Services"],
    "Car Rental & Self-Drive": ["Automotive Rental"],
    "Corporate (Staff & Executive Cars)": ["Information Technology", "Manufacturing", "Pharmaceuticals",
                                           "Banking & Financial Services", "Retail", "Construction & Real Estate"],
    "Leasing & Subscription": ["Banking & Financial Services"],
    "Government / PSU": ["Government"],
    "Hospitals & Education": ["Healthcare", "Education"],
    "Hotels & Travel": ["Hospitality"],
}
INDUSTRY_VARIANTS = {   # non-standard values a CRM accumulates over time
    "Information Technology": ["IT", "I.T.", "Info Tech", "IT Services", "Software"],
    "Banking & Financial Services": ["BFSI", "Banking", "Finance"],
    "Transportation Services": ["Transport", "Travels", "Transportation"],
    "Healthcare": ["Health Care", "Hospital", "Medical"],
    "Hospitality": ["Hotels", "Hotel & Resorts"],
    "Pharmaceuticals": ["Pharma"],
    "Manufacturing": ["Mfg", "Manufacturer"],
    "Construction & Real Estate": ["Real Estate", "Construction"],
    "Automotive Rental": ["Rental", "Car Rental"],
    "Education": ["Edu", "Educational Institution"],
    "Government": ["Govt", "Govt."],
}

CORES = {
    "Employee Transport Services": ["Employee Transport Solutions", "Corporate Mobility", "Staff Transport Services",
                                    "Commute Solutions", "Mobility Services"],
    "Cab Fleet Operator": ["Tours & Travels", "Travels", "Cabs", "Cab Services", "Fleet Services", "Taxi Services"],
    "Car Rental & Self-Drive": ["Car Rentals", "Self Drive Cars", "Rent-a-Car", "Car Hire"],
    "Corporate (Staff & Executive Cars)": {
        "Information Technology": ["Technologies", "Infotech", "Software Solutions", "Digital Systems"],
        "Manufacturing": ["Industries", "Engineering Works", "Precision Components", "Polymers"],
        "Pharmaceuticals": ["Pharmaceuticals", "Life Sciences", "Healthcare Products"],
        "Banking & Financial Services": ["Finance", "Capital", "Credit Services"],
        "Retail": ["Retail", "Supermarts", "Traders"],
        "Construction & Real Estate": ["Builders", "Constructions", "Infra Projects", "Developers"]},
    "Leasing & Subscription": ["Leasing", "Fleet Leasing", "Auto Leasing", "Car Subscription Services"],
    "Hospitals & Education": {
        "Healthcare": ["Hospital", "Multispeciality Hospital", "Medical Trust", "Diagnostics"],
        "Education": ["Institute of Technology", "Public School", "Education Trust", "College"]},
    "Hotels & Travel": ["Hotels & Resorts", "Holidays", "Resorts", "Hospitality Services"],
}
GOVT_DEPTS = ["District Health Office", "Rural Development Office", "Police Motor Transport Unit",
              "Forest Division Office", "Public Works Division", "Zilla Panchayat", "City Municipal Office",
              "District Collectorate", "Agriculture Department Office"]
PREFIXES = ["Kaveri", "Nandi", "Chamundi", "Sahyadri", "Tunga", "Vijaya", "Sri Lakshmi", "Sri Ganesh", "Namma",
            "Deccan", "Hampi", "Malnad", "Garuda", "Airavata", "Suvarna", "Kadamba", "Hoysala", "Vishwa",
            "Pragati", "Surya", "Chandra", "Bharath", "Shree Balaji", "Sai Krupa", "Metro", "Silicon",
            "Garden City", "Karavali", "Akshaya", "Sharada", "Mahalakshmi", "Vinayaka", "Annapoorna", "Srinidhi",
            "Sanjeevini", "Navya", "Omkar", "Trishul", "Pinnacle", "Horizon", "Zenith", "Evergreen", "Bluechip",
            "Greenline", "Urbanline", "Swift", "Rapid", "Orbit", "Crescent", "Sterling", "Vega", "Lotus",
            "Emerald", "Sapphire", "Skyline", "Sunrise", "Heritage", "Royal", "Unity", "Prime", "Elite", "Galaxy",
            "Amrutha", "Brindavan", "Kodagu", "Sharavathi", "Netravati", "Krishna", "Tulasi", "Maruthi Nagar",
            "Anugraha", "Chaitanya", "Samruddhi", "Shubham", "Aadhya", "Vasudha", "Ujwal", "Yashas", "Nakshatra"]

KA_DISTRICTS = ["Bengaluru Urban", "Bengaluru Rural", "Mysuru", "Dharwad", "Dakshina Kannada", "Belagavi",
                "Davanagere", "Shivamogga", "Tumakuru", "Udupi", "Mandya", "Hassan", "Chikkamagaluru", "Kodagu",
                "Ballari", "Kalaburagi", "Vijayapura", "Raichur", "Bidar", "Chitradurga", "Kolar", "Uttara Kannada",
                "Haveri", "Gadag", "Koppal", "Bagalkote", "Chamarajanagar", "Ramanagara", "Chikkaballapur", "Yadgir"]
KA_CITIES = ["Bengaluru", "Mysuru", "Hubballi", "Mangaluru", "Belagavi", "Davanagere", "Shivamogga", "Tumakuru",
             "Udupi"]
KA_CITY_W = [0.60, 0.10, 0.07, 0.08, 0.04, 0.03, 0.03, 0.03, 0.02]
CITY_VARIANTS = {"Bengaluru": ["Bangalore", "BLR", "bengaluru", "Bengaluru "], "Mysuru": ["Mysore"],
                 "Mangaluru": ["Mangalore"], "Hubballi": ["Hubli"], "Belagavi": ["Belgaum"],
                 "Shivamogga": ["Shimoga"], "Tumakuru": ["Tumkur"], "Davanagere": ["Davangere"],
                 "Udupi": ["udupi"]}
OUT_OF_STATE_HQ = [("Maharashtra", "27", "Mumbai"), ("Tamil Nadu", "33", "Chennai"), ("Telangana", "36", "Hyderabad")]

FIRST = ["Aarav", "Abhishek", "Aditi", "Aditya", "Akash", "Akshata", "Amit", "Ananya", "Anil", "Anita", "Anjali",
         "Arjun", "Arun", "Ashwini", "Bharath", "Chaitra", "Chetan", "Deepa", "Deepak", "Divya", "Ganesh",
         "Gautam", "Girish", "Harish", "Harsha", "Kavya", "Keerthi", "Kiran", "Lakshmi", "Madhu", "Mahesh",
         "Manjunath", "Meghana", "Mohan", "Naveen", "Nayana", "Nikhil", "Pallavi", "Pavan", "Pooja", "Prakash",
         "Pradeep", "Prajwal", "Priya", "Raghu", "Rahul", "Rajesh", "Ramesh", "Ravi", "Rekha", "Rohan", "Sachin",
         "Sandeep", "Sanjay", "Shilpa", "Shreya", "Shruti", "Sneha", "Srinivas", "Suma", "Sunil", "Suresh",
         "Swathi", "Tejas", "Uday", "Vani", "Varun", "Vidya", "Vijay", "Vikram", "Vinay", "Vinod", "Imran",
         "Farhan", "Ayesha", "Joseph", "Mary", "Ashok", "Rashmi", "Nandini", "Karthik", "Mithun", "Shashank",
         "Sowmya", "Vasanth", "Yogesh", "Zoya", "Irfan", "Sameer", "Neha", "Ritu", "Manoj", "Asha", "Bhavya"]
LAST = ["Gowda", "Shetty", "Rao", "Hegde", "Kumar", "Reddy", "Naik", "Patil", "Sharma", "Iyer", "Nair",
        "Kulkarni", "Desai", "Joshi", "Singh", "Khan", "D'Souza", "Pinto", "Bhat", "Kamath", "Pai", "Menon",
        "Verma", "Gupta", "Murthy", "Prasad", "Swamy", "Acharya", "Kini", "Shenoy", "Poojary", "Hiremath",
        "Angadi", "Jain", "Agarwal", "Fernandes", "Chandra", "Raju", "Babu", "Krishnan"]
PERSONAL_EMAIL_DOMAINS = ["gmail.com", "gmail.com", "gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com"]
B2B_DESIGNATIONS = ["Fleet Manager", "Admin Head", "Procurement Manager", "Owner / Director", "Transport Manager",
                    "Finance Controller", "Operations Head", "Facilities Manager"]
DECISION_MAKERS = {"Owner / Director", "Procurement Manager", "Finance Controller", "Operations Head"}

RETAIL_SOURCES = {"Walk-in": 0.30, "OEM Website Lead": 0.15, "Dealer Website": 0.08, "Google Ads": 0.10,
                  "Social Media Ads": 0.14, "Car Portal": 0.10, "Referral": 0.08, "Exchange Mela / Event": 0.05}
RETAIL_CONV = {"Walk-in": 0.62, "OEM Website Lead": 0.34, "Dealer Website": 0.36, "Google Ads": 0.27,
               "Social Media Ads": 0.21, "Car Portal": 0.25, "Referral": 0.55, "Exchange Mela / Event": 0.40}
ONLINE_SOURCES = {"OEM Website Lead", "Dealer Website", "Google Ads", "Social Media Ads", "Car Portal"}
NEW_B2B_SOURCES = {"LinkedIn Sales Navigator": 0.22, "Cold Call": 0.20, "Website Inquiry": 0.18,
                   "Corporate Tie-up": 0.12, "Referral": 0.12, "Auto Expo / Event": 0.08, "Tender / RFP": 0.08}
B2B_CONV = {"LinkedIn Sales Navigator": 0.25, "Cold Call": 0.15, "Website Inquiry": 0.35, "Corporate Tie-up": 0.60,
            "Referral": 0.55, "Auto Expo / Event": 0.30, "Tender / RFP": 0.70, "Existing Customer": 0.60,
            "OEM Fleet Desk Referral": 0.60}

STAGES = ["1-Enquiry", "2-Test Drive / Demo", "3-Quotation", "4-Negotiation", "5-Booking / PO",
          "6-Closed Won", "7-Closed Lost"]
STAGE_PROB = [10, 20, 40, 60, 90, 100, 0]
RETAIL_ADV = [0.78, 0.80, 0.78, 0.72, 0.93]      # probability of moving to the next stage
RETAIL_DWELL = [2.5, 3.0, 4.0, 5.0, 18.0]        # mean days in stage (last = booking -> delivery wait)
B2B_ADV = [0.80, 0.72, 0.75, 0.62, 0.96]
B2B_DWELL = [8.0, 12.0, 15.0, 20.0, 28.0]
RETAIL_LOST = ["Price / Discount", "Bought Competitor - Maruti Suzuki", "Bought Competitor - Hyundai",
               "Bought Competitor - Mahindra", "Bought Competitor - Kia", "Loan Rejected", "Waiting Period Too Long",
               "Plan Postponed", "Not Responding"]
RETAIL_LOST_W = [0.20, 0.14, 0.12, 0.10, 0.06, 0.08, 0.08, 0.12, 0.10]
B2B_LOST = ["Pricing", "Competitor Fleet Offer", "Budget Freeze", "Tender Not Awarded", "Delivery Timeline",
            "Went with Leasing Company", "No Decision"]
B2B_LOST_W = [0.24, 0.20, 0.14, 0.08, 0.12, 0.08, 0.14]

SEASON = {1: 1.05, 2: 0.95, 3: 1.25, 4: 0.95, 5: 0.92, 6: 0.85, 7: 0.85, 8: 0.95, 9: 1.05, 10: 1.35, 11: 1.25, 12: 1.0}
WEEKDAY = [0.85, 0.80, 0.85, 0.90, 0.95, 1.35, 1.50]
GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_checksum(first14):
    total = 0
    for i, ch in enumerate(first14):
        p = GST_CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += p // 36 + p % 36
    return GST_CHARS[(36 - total % 36) % 36]


class Generator:
    def __init__(self, args):
        self.rng = np.random.default_rng(args.seed)
        self.seed, self.scale, self.out = args.seed, args.scale, args.out
        as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        self.END = as_of.normalize()
        self.START = self.END - pd.Timedelta(days=int(round(365.25 * args.years))) + pd.Timedelta(days=1)
        self.T = (self.END - self.START).days + 1          # t is "days since START", valid range [0, T)
        self.issues = []                                    # answer key rows

    # ---------------------------------------------------------------- helpers
    def d2t(self, s):
        return (pd.Timestamp(s) - self.START).days

    def years(self, t):
        return np.asarray(t, float) / 365.25

    def frac(self, t):
        return np.clip(np.asarray(t, float) / self.T, 0, 1)

    def fmt(self, t, with_time=True):
        s = self.START + pd.to_timedelta(pd.Series(np.asarray(t, float)), unit="D")
        return s.dt.strftime("%Y-%m-%d %H:%M:%S" if with_time else "%Y-%m-%d").fillna("")

    def pick(self, items, n=None, p=None):
        p = None if p is None else np.asarray(p, float) / np.sum(p)
        idx = self.rng.choice(len(items), size=n, p=p)
        return items[idx] if n is None else [items[i] for i in idx]

    def names(self, n):
        return [f"{a} {b}" for a, b in zip(self.pick(FIRST, n), self.pick(LAST, n))]

    def mobiles(self, n):
        first = self.rng.choice([9, 8, 7, 6], size=n, p=[0.45, 0.25, 0.2, 0.1])
        rest = self.rng.integers(0, 10 ** 9, size=n)
        return [f"{a}{b:09d}" for a, b in zip(first, rest)]

    def issue(self, issue_type, table, rid, related="", detail=""):
        self.issues.append((issue_type, table, rid, related, detail))

    # ---------------------------------------------------------------- users
    def build_users(self):
        rng = self.rng
        self.users, self.slots, self.hold = [], {}, {}

        def add_slot(sid, outlet, role, channel, tenure_m, target=(0, 0)):
            holders, join = [], -rng.uniform(20, 1500)
            while join < self.T:
                ex = join + rng.gamma(2.0, tenure_m * 30.44 / 2.0)
                ex = np.nan if ex >= self.T else ex
                if np.isnan(ex) or ex > 0:
                    idx = len(self.users)
                    self.users.append(dict(idx=idx, role=role, outlet=outlet, channel=channel, join=join, exit=ex,
                                           slot=sid, target=int(rng.integers(*target)) if target[1] else 0))
                    holders.append((idx, join, ex))
                if np.isnan(ex):
                    break
                join = ex + rng.uniform(0, 45)
            self.slots[sid] = dict(role=role, outlet=outlet, holders=holders)
            h = np.full(self.T, -1)
            for idx, j, e in holders:
                d0, d1 = max(0, int(np.ceil(j))), self.T if np.isnan(e) else min(self.T, int(np.floor(e)))
                if d1 > d0:
                    h[d0:d1] = idx
            self.hold[sid] = h

        add_slot("HQ-GM", "HQ", "GM - Sales", "Head Office", 70)
        add_slot("HQ-MIS-1", "HQ", "Sales Ops / MIS Analyst", "Head Office", 26)
        add_slot("HQ-MIS-2", "HQ", "Sales Ops / MIS Analyst", "Head Office", 26)
        self.consultant_slots = {}
        for oid, _, _, _, ch, w in OUTLETS:
            if ch != "Retail":
                continue
            add_slot(f"{oid}-OSM", oid, "Outlet Sales Manager", "Retail", 40)
            add_slot(f"{oid}-TL", oid, "Sales Team Lead", "Retail", 30)
            n = int(round(6 * w)) + 1
            self.consultant_slots[oid] = [f"{oid}-SC-{i}" for i in range(1, n + 1)]
            for s in self.consultant_slots[oid]:
                add_slot(s, oid, "Sales Consultant", "Retail", 16, (8, 15))
        add_slot("OUT08-HEAD", "OUT08", "Head - Corporate & Fleet Sales", "Corporate & Fleet", 60)
        self.kam_slots = [f"OUT08-KAM-{i}" for i in range(1, 4)]
        self.cse_slots = [f"OUT08-CSE-{i}" for i in range(1, 9)]
        for s in self.kam_slots:
            add_slot(s, "OUT08", "Key Account Manager", "Corporate & Fleet", 36, (40, 90))
        for s in self.cse_slots:
            add_slot(s, "OUT08", "Corporate Sales Executive", "Corporate & Fleet", 22, (15, 45))

        def holder_at(sid, t):
            d = int(np.clip(np.floor(t), 0, self.T - 1))
            return self.hold[sid][d]

        chain = {"Sales Consultant": lambda u: [f"{u['outlet']}-TL", f"{u['outlet']}-OSM", "HQ-GM"],
                 "Sales Team Lead": lambda u: [f"{u['outlet']}-OSM", "HQ-GM"],
                 "Outlet Sales Manager": lambda u: ["HQ-GM"],
                 "Corporate Sales Executive": lambda u: ["OUT08-HEAD", "HQ-GM"],
                 "Key Account Manager": lambda u: ["OUT08-HEAD", "HQ-GM"],
                 "Head - Corporate & Fleet Sales": lambda u: ["HQ-GM"],
                 "Sales Ops / MIS Analyst": lambda u: ["HQ-GM"], "GM - Sales": lambda u: []}
        names = self.names(len(self.users))
        rows = []
        for u, nm in zip(self.users, names):
            mgr = -1
            for s in chain[u["role"]](u):
                mgr = holder_at(s, max(u["join"], 0) + 1)
                if mgr >= 0:
                    break
            u["name"] = nm
            first, last = nm.lower().replace("'", "").split(" ", 1)
            rows.append(dict(UserID=f"USR-{u['idx'] + 1:03d}", FullName=nm, Role=u["role"], Channel=u["channel"],
                             OutletID=u["outlet"], ManagerUserID=f"USR-{mgr + 1:03d}" if mgr >= 0 else "",
                             Email=f"{first}.{last.replace(' ', '')}{u['idx'] + 1}@{STAFF_EMAIL_DOMAIN}",
                             JoinDate=None, ExitDate=None, IsActive=np.isnan(u["exit"]),
                             MonthlyUnitTarget=u["target"] or ""))
        df = pd.DataFrame(rows)
        df["JoinDate"] = self.fmt([u["join"] for u in self.users], False).values
        df["ExitDate"] = self.fmt([u["exit"] for u in self.users], False).values
        self.df_users = df
        self.user_exit = np.array([u["exit"] for u in self.users])

    def uid(self, idx_arr):
        return [f"USR-{i + 1:03d}" if i >= 0 else "" for i in np.asarray(idx_arr)]

    def assign(self, slot_ids, t_arr, fallback=()):
        t_arr = np.asarray(t_arr, float)
        days = np.clip(np.floor(t_arr).astype(int), 0, self.T - 1)
        H = np.stack([self.hold[s] for s in slot_ids])
        res = H[self.rng.integers(0, len(slot_ids), len(days)), days]
        miss = res < 0
        if miss.any():
            res[miss] = H[self.rng.integers(0, len(slot_ids), miss.sum()), days[miss]]
        for s in fallback:
            miss = res < 0
            if not miss.any():
                break
            res[miss] = self.hold[s][days[miss]]
        return res

    def successor(self, uidx):
        """Next person to hold the same slot after this user left (-1 if nobody yet)."""
        u = self.users[uidx]
        holders = self.slots[u["slot"]]["holders"]
        for k, (idx, j, e) in enumerate(holders):
            if idx == uidx:
                return holders[k + 1][0] if k + 1 < len(holders) else -1
        return -1

    def is_active(self, uidx, t):
        return uidx >= 0 and (np.isnan(self.user_exit[uidx]) or self.user_exit[uidx] > t)

    # ---------------------------------------------------------------- products
    def build_products(self):
        df = pd.DataFrame(PRODUCTS, columns=["ProductID", "Model", "Powertrain", "Segment", "FuelType",
                                             "PriceLakh", "LaunchDate", "FleetOnly", "RetailPop"])
        df["IsEV"] = df["FuelType"].eq("Electric")
        df["IndicativeExShowroomINR"] = (df["PriceLakh"] * 1e5).astype(int)
        self.prod = df
        self.prod_launch_t = np.array([self.d2t(d) for d in df["LaunchDate"]], float)
        self.prod_price = df.set_index("ProductID")["IndicativeExShowroomINR"].to_dict()
        self.prod_ev = df.set_index("ProductID")["IsEV"].to_dict()
        self.prod_label = (df["Model"] + " " + np.where(df["Powertrain"] == "EV", "", df["Powertrain"])).str.strip()
        self.prod_label = dict(zip(df["ProductID"], self.prod_label))
        self.df_products = df[["ProductID", "Model", "Powertrain", "Segment", "FuelType", "IsEV", "FleetOnly",
                               "LaunchDate", "IndicativeExShowroomINR"]].rename(columns={"FleetOnly": "IsFleetOnly"})
        self.df_products["IsFleetOnly"] = self.df_products["IsFleetOnly"].astype(bool)

    def retail_products(self, t_arr):
        t_arr = np.asarray(t_arr, float)
        base = self.prod["RetailPop"].values.astype(float) * (self.prod["FleetOnly"].values == 0)
        ev = self.prod["IsEV"].values
        w = base[None, :] * (self.prod_launch_t[None, :] <= t_arr[:, None])
        w = w * np.where(ev[None, :], 1 + 0.9 * self.frac(t_arr)[:, None], 1.0)
        cw = np.cumsum(w, axis=1)
        idx = (cw < self.rng.random(len(t_arr))[:, None] * cw[:, -1:]).sum(axis=1)
        return self.prod["ProductID"].values[idx]

    def b2b_product(self, atype, t):
        mix = ACCT_TYPES[atype]["mix"]
        ids = [p for p in mix if self.prod_launch_t[int(p[3:]) - 1] <= t]
        w = np.array([mix[p] * (1 + 0.8 * self.frac(t) if self.prod_ev[p] else 1.0) for p in ids])
        return ids[self.rng.choice(len(ids), p=w / w.sum())]

    # ---------------------------------------------------------------- accounts
    def make_pan(self, entity, name):
        L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        r = self.rng
        initial = next((c for c in name.upper() if c.isalpha()), "A")
        return ("".join(r.choice(list(L), 3)) + entity + initial + f"{r.integers(0, 10000):04d}" + r.choice(list(L)))

    def make_gstin(self, state_code, pan, entity_no="1"):
        s14 = f"{state_code}{pan}{entity_no}Z"
        return s14 + gstin_checksum(s14)

    def new_duns(self):
        while True:
            d = str(self.rng.integers(100_000_000, 999_999_999))
            if d not in self.used_duns:
                self.used_duns.add(d)
                return d

    def new_account_name(self, atype, prefix=None):
        r = self.rng
        for attempt in range(40):
            industry = self.pick(INDUSTRY_BY_TYPE[atype])
            if atype == "Government / PSU":
                name = f"{self.pick(GOVT_DEPTS)} - {self.pick(KA_DISTRICTS)}"
                if attempt > 20:
                    name += f" (Unit {r.integers(2, 9)})"
                entity, suffix = "G", ""
            else:
                p = prefix or self.pick(PREFIXES)
                cores = CORES[atype]
                core = self.pick(cores[industry] if isinstance(cores, dict) else cores)
                if atype in ("Cab Fleet Operator", "Car Rental & Self-Drive", "Hotels & Travel"):
                    suffix, entity = [("Pvt Ltd", "C"), ("LLP", "F"), ("", "P")][r.choice(3, p=[0.45, 0.15, 0.40])]
                elif atype == "Hospitals & Education":
                    suffix, entity = [("Pvt Ltd", "C"), ("", "T")][r.choice(2, p=[0.4, 0.6])]
                else:
                    suffix, entity = [("Pvt Ltd", "C"), ("Ltd", "C"), ("LLP", "F")][r.choice(3, p=[0.75, 0.1, 0.15])]
                name = f"{p} {core}" + (f" {suffix}" if suffix else "")
                if attempt > 20:
                    name = f"{p} {core} {self.pick(KA_CITIES)}" + (f" {suffix}" if suffix else "")
            if name not in self.used_names:
                self.used_names.add(name)
                slug = "".join(ch for ch in name.split(" Pvt")[0].split(" LLP")[0].lower() if ch.isalnum())[:22]
                domain = "" if atype == "Government / PSU" else slug + self.pick([".in", ".com", ".co.in"])
                return dict(name=name, industry=industry, entity=entity, domain=domain,
                            prefix=(prefix or name.split(" ")[0]))
        raise RuntimeError("could not create a unique account name")

    def build_accounts(self):
        r, sc = self.rng, self.scale
        self.used_names, self.used_duns = set(), set()
        types = list(ACCT_TYPES)
        shares = [ACCT_TYPES[t]["share"] for t in types]
        acc = []

        def base_account(atype, created, parent=None, prefix=None, is_parent=False):
            nm = self.new_account_name(atype, prefix)
            if is_parent and atype in ("Employee Transport Services", "Leasing & Subscription") and r.random() < 0.3:
                state, code, city = OUT_OF_STATE_HQ[r.integers(0, 3)]
            else:
                state, code, city = "Karnataka", "29", self.pick(KA_CITIES, p=KA_CITY_W)
            pan = self.make_pan(nm["entity"], nm["name"])
            fleet_med = ACCT_TYPES[atype]["fleet"] * (2.5 if is_parent else 1.0)
            return dict(name=nm["name"], atype=atype, industry=nm["industry"], entity=nm["entity"],
                        domain=nm["domain"], state=state, city=city, pan=pan, gstin=self.make_gstin(code, pan),
                        duns="", parent_duns="", parent=parent, created=created, is_parent=is_parent,
                        fleet=int(max(1, round(r.lognormal(np.log(fleet_med), 0.7)))), kn="", dup_of=None,
                        is_branch=False)

        # parent groups with branches / group companies
        n_groups = int(170 * sc)
        for _ in range(n_groups):
            atype = self.pick(types, p=[s * (2.0 if t in ("Employee Transport Services", "Leasing & Subscription",
                                                          "Corporate (Staff & Executive Cars)") else 1.0)
                                        for t, s in zip(types, shares)])
            if atype == "Government / PSU":
                atype = "Corporate (Staff & Executive Cars)"
            created = -r.uniform(30, 5 * 365) if r.random() < 0.8 else r.uniform(0, self.T * 0.6)
            p = base_account(atype, created, is_parent=True)
            p["kn"] = "Key" if (atype in ("Employee Transport Services", "Leasing & Subscription",
                                          "Corporate (Staff & Executive Cars)") and r.random() < 0.55) \
                else ("Key" if r.random() < 0.25 else "Named")
            if r.random() < min(0.95, ACCT_TYPES[atype]["duns"] + 0.25):
                p["duns"] = self.new_duns()
            pi = len(acc)
            acc.append(p)
            for _ in range(int(r.integers(2, 7))):
                c_created = min(p["created"] + r.uniform(30, 900), self.T - 1)
                if r.random() < 0.6:        # branch of the same legal entity
                    city = self.pick([c for c in KA_CITIES if c != p["city"]])
                    c = dict(p)
                    c.update(name=f"{p['name']} - {city} Branch", city=city, state="Karnataka", created=c_created,
                             parent=pi, is_parent=False, is_branch=True, duns="", parent_duns="",
                             fleet=int(max(1, round(p["fleet"] * r.uniform(0.1, 0.35)))))
                    c["gstin"] = p["gstin"] if p["state"] == "Karnataka" else self.make_gstin("29", p["pan"])
                    self.used_names.add(c["name"])
                else:                       # separate legal entity in the same group
                    c = base_account(atype, c_created, parent=pi, prefix=p["name"].split(" ")[0])
                    if r.random() < 0.5:
                        c["domain"] = p["domain"]
                c["kn"] = p["kn"] if r.random() < 0.8 else ""
                if p["duns"] and r.random() < 0.7:
                    c["duns"], c["parent_duns"] = self.new_duns(), p["duns"]
                acc.append(c)

        # standalone accounts
        for _ in range(int(3900 * sc)):
            atype = self.pick(types, p=shares)
            created = -r.uniform(1, 5 * 365) if r.random() < 0.55 else self.T * np.sqrt(r.random())
            a = base_account(atype, created)
            a["kn"] = "Key" if r.random() < 0.03 else ("Named" if r.random() < 0.15 else "")
            if r.random() < ACCT_TYPES[atype]["duns"]:
                a["duns"] = self.new_duns()
            acc.append(a)

        n_base = len(acc)
        for i, a in enumerate(acc):
            a["id"] = f"ACC-{i + 1:05d}"

        # owners (Key accounts -> KAMs, others -> Corporate Sales Executives)
        created = np.array([max(a["created"], 0) for a in acc])
        is_key = np.array([a["kn"] == "Key" for a in acc])
        owners = np.where(is_key, self.assign(self.kam_slots, created, ["OUT08-HEAD"]),
                          self.assign(self.cse_slots, created, ["OUT08-HEAD"]))
        for a, o in zip(acc, owners):
            a["owner"] = int(o)
            # when an owner leaves, the manager usually hands their book to the successor (75% of the time)
            while a["owner"] >= 0 and not np.isnan(self.user_exit[a["owner"]]) and r.random() < 0.75:
                succ = self.successor(a["owner"])
                if succ < 0:
                    break
                a["owner"] = succ

        # ---- planted duplicates
        dup_candidates = [i for i, a in enumerate(acc) if not a["is_branch"]]
        n_dup = int(len(acc) * 0.04)
        for oi in r.choice(dup_candidates, size=n_dup, replace=False):
            o = acc[oi]
            d = dict(o)
            d_created = r.uniform(max(o["created"], 0) + 30, self.T - 1) if max(o["created"], 0) + 30 < self.T - 1 \
                else self.T - 2
            roll = r.random()
            gst = o["gstin"] if roll < 0.55 else ("" if roll < 0.75 else self.malform_gstin(o["gstin"]))
            d.update(id=f"ACC-{len(acc) + 1:05d}", name=self.name_variant(o["name"]), created=d_created,
                     gstin=gst, domain=o["domain"] if r.random() < 0.75 else "", parent=None, is_parent=False,
                     duns=o["duns"] if (o["duns"] and r.random() < 0.2) else "", parent_duns="", kn="",
                     dup_of=int(oi), owner=int(self.assign(self.cse_slots, [d_created], ["OUT08-HEAD"])[0]))
            if r.random() < 0.3:
                d["city"] = self.pick(CITY_VARIANTS.get(o["city"], [o["city"]]))
            acc.append(d)
            self.issue("duplicate_account", "dim_accounts", d["id"], o["id"],
                       "Same business as the related account; created again with a variant name")

        # ---- other planted account issues
        for i, a in enumerate(acc):
            if a["dup_of"] is not None:
                continue
            u = r.random()
            if u < 0.08:
                a["gstin"] = ""
                self.issue("missing_gstin", "dim_accounts", a["id"])
            elif u < 0.11:
                a["gstin"] = self.malform_gstin(a["gstin"])
                self.issue("malformed_gstin", "dim_accounts", a["id"], detail=a["gstin"])
            std = a["industry"]
            v = r.random()
            if v < 0.09 and std in INDUSTRY_VARIANTS:
                a["industry"] = self.pick(INDUSTRY_VARIANTS[std])
                self.issue("nonstandard_industry", "dim_accounts", a["id"], detail=f"{a['industry']} -> {std}")
            elif v < 0.11:
                a["industry"] = ""
                self.issue("missing_industry", "dim_accounts", a["id"])
            if a["city"] in CITY_VARIANTS and r.random() < 0.07:
                std_city = a["city"]
                a["city"] = self.pick(CITY_VARIANTS[std_city])
                self.issue("nonstandard_city", "dim_accounts", a["id"], detail=f"'{a['city']}' -> {std_city}")
        # hierarchy problems (children only)
        parent_idx = [i for i, a in enumerate(acc) if a["is_parent"]]
        for i, a in enumerate(acc):
            if a["parent"] is None or a["dup_of"] is not None:
                continue
            u = r.random()
            true_parent = acc[a["parent"]]["id"]
            if u < 0.06:
                a["parent"] = None
                self.issue("missing_parent_link", "dim_accounts", a["id"], true_parent,
                           "ParentAccountID blank; name / D&B ParentDUNS point to the group")
            elif u < 0.075:
                a["parent"] = "ORPHAN"
                self.issue("orphan_parent_reference", "dim_accounts", a["id"], detail="ParentAccountID does not exist")
            elif u < 0.14 and a["parent_duns"]:
                wrong = int(r.choice(parent_idx))
                if acc[wrong]["id"] != true_parent:
                    a["parent"] = wrong
                    self.issue("parent_misaligned_with_dnb", "dim_accounts", a["id"], true_parent,
                               "CRM parent differs from the D&B family tree (ParentDUNS)")
        for i, a in enumerate(acc):
            if isinstance(a["parent"], int) and a["dup_of"] is None and not a["is_parent"]:
                p = acc[a["parent"]]
                if p["kn"] and a["kn"] != p["kn"]:
                    self.issue("kn_designation_inconsistent", "dim_accounts", a["id"], p["id"],
                               f"Child '{a['kn'] or 'blank'}' vs parent '{p['kn']}'")

        self.acc = acc
        rows = []
        for a in acc:
            par = a["parent"]
            par_id = "" if par is None else ("ACC-9" + a["id"][5:] if par == "ORPHAN" else acc[par]["id"])
            rows.append(dict(AccountID=a["id"], AccountName=a["name"], AccountType=a["atype"],
                             Industry=a["industry"], ParentAccountID=par_id, City=a["city"], State=a["state"],
                             Website=a["domain"], GSTIN=a["gstin"], DUNS=a["duns"], ParentDUNS=a["parent_duns"],
                             KN_Designation=a["kn"], EstimatedFleetSize=a["fleet"],
                             OwnerUserID=self.uid([a["owner"]])[0], CreatedOn=a["created"]))
        df = pd.DataFrame(rows)
        df["CreatedOn"] = self.fmt(df["CreatedOn"]).values
        self.df_accounts = df
        self.acc_index = {a["id"]: i for i, a in enumerate(acc)}

    def malform_gstin(self, g):
        if not g:
            return g
        k = self.rng.integers(0, 5)
        return [g[:-1], g.lower(), g[:7] + " " + g[7:], g[:13] + "2" + g[14:],
                g[:-1] + GST_CHARS[(GST_CHARS.index(g[-1]) + 7) % 36]][k]

    def name_variant(self, name):
        r = self.rng
        ops = [lambda s: s.upper(),
               lambda s: s.replace(" Pvt Ltd", "").replace(" LLP", "").replace(" Ltd", ""),
               lambda s: s.replace("Pvt Ltd", "Private Limited") if "Pvt Ltd" in s else s + " Pvt. Ltd.",
               lambda s: s.replace("&", "and") if "&" in s else "M/s " + s,
               lambda s: s.replace("Sri ", "Shri ").replace("Shree ", "Sri ") if ("Sri " in s or "Shree " in s)
               else s.replace(" ", "  ", 1),
               lambda s: self.typo(s),
               lambda s: s + "."]
        out = name
        for k in r.choice(len(ops), size=int(r.integers(1, 3)), replace=False):
            out = ops[k](out)
        return out if out != name else name + " "

    def typo(self, s):
        letters = [i for i in range(1, len(s) - 2) if s[i].isalpha() and s[i + 1].isalpha()]
        if not letters:
            return s
        i = int(self.rng.choice(letters))
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]

    # ---------------------------------------------------------------- B2B contacts
    def build_b2b_contacts(self):
        r = self.rng
        rows = []
        self.acc_contacts = {}
        for i, a in enumerate(self.acc):
            if a["dup_of"] is not None:
                continue
            n = 1 + r.poisson(2.2 if a["is_parent"] else 1.0)
            if a["atype"] == "Government / PSU":
                n = min(n, 2)
            nms, mobs = self.names(n), self.mobiles(n)
            for nm, mob in zip(nms, mobs):
                desig = self.pick(B2B_DESIGNATIONS)
                first, last = nm.lower().replace("'", "").split(" ", 1)
                if a["domain"]:
                    email = self.pick([f"{first}.{last}@{a['domain']}", f"{first}@{a['domain']}"])
                else:
                    email = f"{first}{last}{r.integers(10, 99)}@{self.pick(PERSONAL_EMAIL_DOMAINS)}"
                cid = f"CON-{len(rows) + 1:06d}"
                rows.append(dict(ContactID=cid, CustomerType="Business", AccountID=a["id"], FullName=nm,
                                 Designation=desig, IsDecisionMaker=desig in DECISION_MAKERS, Mobile=mob,
                                 Email=email.replace(" ", ""), City=a["city"],
                                 OwnerUserID=self.uid([a["owner"]])[0],
                                 CreatedOn=min(max(a["created"], -3000) + r.uniform(0, 120), self.T - 1)))
                self.acc_contacts.setdefault(i, []).append(len(rows) - 1)
        # duplicate accounts get a copy of one of the original's contacts
        for i, a in enumerate(self.acc):
            if a["dup_of"] is None:
                continue
            src = rows[self.acc_contacts[a["dup_of"]][0]]
            c = dict(src)
            c.update(ContactID=f"CON-{len(rows) + 1:06d}", AccountID=a["id"], FullName=self.person_variant(src["FullName"]),
                     Email="" if r.random() < 0.4 else src["Email"], CreatedOn=a["created"],
                     OwnerUserID=self.uid([a["owner"]])[0])
            rows.append(c)
            self.acc_contacts.setdefault(i, []).append(len(rows) - 1)
            self.issue("duplicate_contact", "dim_contacts", c["ContactID"], src["ContactID"],
                       "Same person, attached to a duplicate account")
        self.contact_rows = rows

    def person_variant(self, nm):
        first, last = nm.split(" ", 1)
        return self.pick([f"{first} {last[0]}", f"Mr. {nm}" if self.rng.random() < 0.5 else f"Ms. {nm}",
                          f"{first.upper()} {last.upper()}", f"{last} {first}", nm.lower()])

    # ---------------------------------------------------------------- leads
    def build_leads(self):
        r, T = self.rng, self.T
        days = np.arange(T)
        dates = self.START + pd.to_timedelta(days, unit="D")
        season = np.array([SEASON[m] for m in dates.month])
        wd = np.array(WEEKDAY)[dates.dayofweek]
        growth = 1 + 0.07 * self.years(days)
        leads = []

        # ---- retail
        src_names = list(RETAIL_SOURCES)
        src_w = np.array(list(RETAIL_SOURCES.values()))
        online_hour_w = np.array([1, .5, .3, .2, .2, .3, .6, 1.2, 2, 3, 3.5, 4, 4, 3.5, 3.5, 3.5, 3.5, 4, 4.5, 5, 5,
                                  4.5, 3.5, 2])
        for oid, _, city, _, ch, w in OUTLETS:
            if ch != "Retail":
                continue
            lam = 11.0 * self.scale * w * season * wd * growth
            cnt = r.poisson(lam)
            d = np.repeat(days, cnt)
            n = len(d)
            src = np.array(src_names)[r.choice(len(src_names), size=n, p=src_w / src_w.sum())]
            online = np.isin(src, list(ONLINE_SOURCES))
            hour = np.where(online, r.choice(24, size=n, p=online_hour_w / online_hour_w.sum()) + r.random(n),
                            r.uniform(10, 19.5, n))
            t = d + hour / 24
            # first response
            delay = np.zeros(n)
            in_hours = (hour >= 9.5) & (hour < 19)
            mins = r.lognormal(np.log(30), 1.0, n)
            wait = np.where(hour >= 19, 24 - hour + 9.5, np.maximum(9.5 - hour, 0))
            delay = np.where(online, np.where(in_hours, mins / 60, wait + r.lognormal(np.log(30), 0.8, n) / 60), 0)
            delay = np.where(src == "Referral", r.lognormal(np.log(90), 0.9, n) / 60, delay)
            never = online & (r.random(n) < 0.05)
            fr = np.where(never, np.nan, t + delay / 24)
            fr = np.where(fr >= T, np.nan, fr)
            owner = self.assign(self.consultant_slots[oid], t, [f"{oid}-TL", f"{oid}-OSM"])
            unassigned = online & (r.random(n) < 0.015)
            owner = np.where(unassigned, -1, owner)
            prod = self.retail_products(t)
            conv_p = np.array([RETAIL_CONV[s] for s in src])
            for k in range(n):
                leads.append(dict(t=t[k], channel="Retail", source=src[k], outlet=oid, owner=int(owner[k]),
                                  fr=fr[k], prod=prod[k], acct=None, company="", city=city, conv_p=conv_p[k],
                                  walkin=src[k] in ("Walk-in", "Exchange Mela / Event"), repeat=False))

        # ---- B2B: origin leads for accounts created in-period, repeat enquiries, new-company enquiries
        for i, a in enumerate(self.acc):
            if a["dup_of"] is not None:
                continue
            at = ACCT_TYPES[a["atype"]]
            if a["created"] >= 0:
                t0 = max(0.0, a["created"] - r.uniform(1, 20))
                src = self.pick(list(NEW_B2B_SOURCES), p=list(NEW_B2B_SOURCES.values()))
                if a["atype"] == "Government / PSU":
                    src = "Tender / RFP"
                leads.append(self.b2b_lead(t0, i, src, force_conv_at=a["created"]))
            start = max(a["created"], 0)
            mult = 2.0 if a["kn"] == "Key" else (1.4 if a["kn"] == "Named" else 1.0)
            k = r.poisson(at["repeat"] * mult * (T - start) / 365.25)
            for t0 in np.sort(r.uniform(start + 5, T, size=k)) if start + 5 < T else []:
                src = self.pick(["Existing Customer", "Referral", "OEM Fleet Desk Referral", "Tender / RFP"],
                                p=[0.6, 0.15, 0.15, 0.10 if a["atype"] != "Government / PSU" else 0.5])
                leads.append(self.b2b_lead(t0, i, src, repeat=True))
        # new companies that never became accounts; some are really existing accounts (unlinked)
        non_dup = [i for i, a in enumerate(self.acc) if a["dup_of"] is None and a["created"] < T - 60]
        for _ in range(int(2200 * self.scale)):
            t0 = T * np.sqrt(r.random())
            src = self.pick(list(NEW_B2B_SOURCES), p=list(NEW_B2B_SOURCES.values()))
            L = self.b2b_lead(t0, None, src)
            L["conv_p"] = 0.0
            if r.random() < 0.06:
                j = int(r.choice(non_dup))
                if self.acc[j]["created"] < t0:
                    L["company"] = self.name_variant(self.acc[j]["name"])
                    L["unlinked_acct"] = self.acc[j]["id"]
            if not L["company"]:
                L["company"] = self.new_account_name(self.pick(list(ACCT_TYPES)))["name"]
            leads.append(L)

        leads.sort(key=lambda x: x["t"])
        self.leads = leads

    def b2b_lead(self, t0, acc_i, src, force_conv_at=None, repeat=False):
        r = self.rng
        fr = t0 + r.lognormal(np.log(6), 0.9) / 24
        a = self.acc[acc_i] if acc_i is not None else None
        if a is not None and self.is_active(a["owner"], t0):
            owner = a["owner"]
        else:
            slots = self.kam_slots if (a is not None and a["kn"] == "Key") else self.cse_slots
            owner = int(self.assign(slots, [t0], ["OUT08-HEAD"])[0])
        prod = self.b2b_product(a["atype"], t0) if a is not None else self.b2b_product(self.pick(list(ACCT_TYPES)), t0)
        return dict(t=t0, channel="Corporate & Fleet", source=src, outlet="OUT08", owner=owner,
                    fr=fr if fr < self.T else np.nan, prod=prod, acct=acc_i,
                    company=a["name"] if a is not None else "", city=a["city"] if a is not None else "",
                    conv_p=1.0 if force_conv_at is not None else B2B_CONV.get(src, 0.4), walkin=False,
                    force_conv_at=force_conv_at, repeat=repeat)

    def finalize_leads(self):
        """Decide lead outcomes, build fact_leads rows and the list of opportunities to simulate."""
        r, T = self.rng, self.T
        rows, opps_to_make = [], []
        retail_people = {}
        for k, L in enumerate(self.leads):
            lid = f"LD-{k + 1:07d}"
            L["id"] = lid
            t, fr = L["t"], L["fr"]
            age = T - t
            converted_t = np.nan
            if not np.isnan(fr) and r.random() < L["conv_p"]:
                if L.get("force_conv_at") is not None:
                    converted_t = max(L["force_conv_at"], fr)
                elif L["channel"] == "Retail":
                    converted_t = fr + (r.uniform(0, 0.3) if (L["walkin"] and r.random() < 0.5) else r.uniform(0.2, 6))
                else:
                    converted_t = fr + r.uniform(1, 21)
                if converted_t >= T:
                    converted_t = np.nan
            status, reason, last = "", "", np.nan
            if not np.isnan(converted_t):
                status, last = "Converted", converted_t
            elif np.isnan(fr):
                status = "New"
            elif age < 30 or r.random() < 0.10:
                status = self.pick(["Contacted", "Qualified"], p=[0.7, 0.3])
                last = min(fr + r.uniform(0, min(age, 12)), T - 0.01) if age < 30 else fr + r.uniform(0, 10)
            elif r.random() < 0.6:
                status = "Lost"
                reason = self.pick(["Not Interested", "Budget Constraint", "Bought Competitor", "Just Enquiring",
                                    "Location Too Far"] if L["channel"] == "Retail" else
                                   ["Not Interested", "Budget Freeze", "Chose Competitor", "No Requirement Now"])
                last = min(fr + r.uniform(0, 20), T - 0.01)
            else:
                status, reason = "Unresponsive", "No Answer After Multiple Attempts"
                last = min(fr + r.uniform(3, 30), T - 0.01)
            L.update(status=status, last=last, conv=converted_t)
            if L["channel"] == "Retail":
                nm = self.names(1)[0]
                mob = self.mobiles(1)[0]
                first, lastn = nm.lower().replace("'", "").split(" ", 1)
                email = f"{first}.{lastn}{r.integers(1, 999)}@{self.pick(PERSONAL_EMAIL_DOMAINS)}" \
                    if r.random() < 0.45 else ""
                L.update(person=nm, mobile=mob, email=email.replace(" ", ""))
            else:
                a = self.acc[L["acct"]] if L["acct"] is not None else None
                L.update(person="", mobile="", email="")
            if not np.isnan(converted_t):
                opps_to_make.append(k)
            if L.get("unlinked_acct"):
                self.issue("b2b_lead_not_linked_to_existing_account", "fact_leads", lid, L["unlinked_acct"],
                           f"Company '{L['company']}' already exists as an account")
            if L["owner"] < 0:
                self.issue("lead_unassigned", "fact_leads", lid)
            rows.append(L)

        # planted duplicate retail enquiries (same person, another source, within 10 days)
        retail_idx = [k for k, L in enumerate(rows) if L["channel"] == "Retail"]
        extra = []
        for k in r.choice(retail_idx, size=int(len(retail_idx) * 0.03), replace=False):
            L = rows[k]
            t2 = L["t"] + r.uniform(0.2, 10)
            if t2 >= T:
                continue
            d = dict(L)
            d.update(t=t2, source=self.pick([s for s in RETAIL_SOURCES if s != L["source"]]),
                     owner=int(self.assign(self.consultant_slots[L["outlet"]], [t2])[0]),
                     fr=min(t2 + r.uniform(0.01, 0.5), T - 0.01), status=self.pick(["Contacted", "Lost"]),
                     conv=np.nan, dup_of=L["id"])
            d["last"] = min(d["fr"] + r.uniform(0, 3), T - 0.01)
            extra.append(d)
        base_n = len(rows)
        for j, d in enumerate(extra):
            d["id"] = f"LD-{base_n + j + 1:07d}"
            self.issue("duplicate_lead", "fact_leads", d["id"], d["dup_of"], "Same customer enquired again via another source")
        self.lead_rows = rows + extra
        self.opps_to_make = opps_to_make

    # ---------------------------------------------------------------- opportunities
    def build_opportunities(self):
        r, T = self.rng, self.T
        opps, hist, lines = [], [], []
        contacts = self.contact_rows
        for n, k in enumerate(self.opps_to_make):
            L = self.lead_rows[k]
            oid = f"OPP-{n + 1:06d}"
            L["opp_id"] = oid
            t0 = L["conv"]
            retail = L["channel"] == "Retail"
            a = self.acc[L["acct"]] if L["acct"] is not None else None
            tender = L["source"] == "Tender / RFP"
            # owner
            if retail:
                owner = L["owner"] if L["owner"] >= 0 else int(
                    self.assign(self.consultant_slots[L["outlet"]], [t0], [f"{L['outlet']}-TL"])[0])
            else:
                owner = L["owner"]
            # contact
            if retail:
                cid = f"CON-{len(contacts) + 1:06d}"
                contacts.append(dict(ContactID=cid, CustomerType="Individual", AccountID="", FullName=L["person"],
                                     Designation="", IsDecisionMaker=True, Mobile=L["mobile"], Email=L["email"],
                                     City=L["city"], OwnerUserID=self.uid([owner])[0], CreatedOn=t0))
            else:
                cands = self.acc_contacts.get(L["acct"], [])
                dm = [c for c in cands if contacts[c]["IsDecisionMaker"]]
                cid = contacts[int(r.choice(dm or cands))]["ContactID"] if cands else ""
            # stage simulation
            adv, dwell = (RETAIL_ADV, RETAIL_DWELL) if retail else (B2B_ADV, B2B_DWELL)
            ev = self.prod_ev[L["prod"]]
            zombie = (t0 < T - 90) and r.random() < 0.035
            zombie_stop = int(r.integers(0, 4)) if zombie else 99
            t, s, status, close_t, lost_reason = t0, 0, "Open", np.nan, ""
            while True:
                mean = dwell[s] * (1.8 if (tender and not retail) else 1.0) + (10 if (s == 4 and ev) else 0)
                dt = r.gamma(2.0, mean / 2.0)
                if s == zombie_stop:
                    hist.append((oid, s, t, np.nan))
                    last = min(t + r.uniform(0, 10), T - 0.01)
                    break
                if t + dt >= T:
                    hist.append((oid, s, t, np.nan))
                    last = max(t, T - r.uniform(0.2, 12))
                    break
                if r.random() < adv[s]:
                    hist.append((oid, s, t, t + dt))
                    t, s = t + dt, s + 1
                    if s == 5:
                        status, close_t = "Won", t
                        hist.append((oid, 5, t, np.nan))
                        last = t
                        break
                else:
                    t = t + dt
                    hist.append((oid, s, hist[-1][2] if hist and hist[-1][0] == oid and np.isnan(hist[-1][3])
                                 else t - dt, t))
                    status, close_t = "Lost", t
                    hist.append((oid, 6, t, np.nan))
                    lost_reason = self.pick(RETAIL_LOST, p=RETAIL_LOST_W) if retail else self.pick(B2B_LOST, p=B2B_LOST_W)
                    last = t
                    break
            if zombie and status == "Open":
                self.issue("abandoned_open_opportunity", "fact_opportunities", oid,
                           detail="Deal went cold but was never closed")
            stage_no = s if status == "Open" else (5 if status == "Won" else 6)
            exp_close = t0 + (25 if retail else (120 if tender else 75)) * r.uniform(0.7, 1.4)
            # product lines
            yrs = self.years(t0)
            month = (self.START + pd.Timedelta(days=float(t0))).month
            if retail:
                prods = [L["prod"] if r.random() < 0.8 else self.retail_products([t0])[0]]
                qtys = [2 if r.random() < 0.03 else 1]
                disc_base = r.uniform(1.5, 5.5) + (2 if month in (10, 11) else 0) + (2 if month == 3 else 0)
            else:
                nl = r.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
                prods = [L["prod"]] + [self.b2b_product(a["atype"], t0) for _ in range(nl - 1)]
                at = ACCT_TYPES[a["atype"]]
                qtys = [int(np.clip(round(r.lognormal(np.log(at["qty"]), 0.8)), 1, at["qmax"])) for _ in prods]
                disc_base = r.uniform(4, 9)
            no_lines = r.random() < 0.01
            total_units, total_value = 0, 0.0
            if not no_lines:
                for p, q in zip(prods, qtys):
                    unit = round(self.prod_price[p] * (1 + 0.03 * yrs) * r.uniform(0.92, 1.20), -2)
                    disc = disc_base + (0 if retail else (2 if q >= 25 else 0) + (3 if q >= 100 else 0))
                    val = round(q * unit * (1 - disc / 100))
                    lines.append(dict(LineID=f"OPL-{len(lines) + 1:07d}", OpportunityID=oid, ProductID=p, Quantity=q,
                                      UnitPriceINR=int(unit), DiscountPct=round(disc, 2), LineValueINR=int(val)))
                    total_units += q
                    total_value += val
            else:
                self.issue("opportunity_missing_products", "fact_opportunities", oid,
                           detail="No product lines, so revenue is 0")
            if retail:
                name = f"{self.prod_label[prods[0]]} - {L['person']}"
            else:
                name = f"{a['name'][:40]} - {total_units or '?'}x {self.prod_label[prods[0]]}"
            fin = self.pick(["Cash", "Bank Loan", "OEM Partner Finance"], p=[0.25, 0.6, 0.15]) if retail else \
                self.pick(["Corporate Purchase", "Leasing", "Bank Loan"], p=[0.4, 0.25, 0.35])
            opps.append(dict(OpportunityID=oid, OpportunityName=name, LeadID=L["id"], Channel=L["channel"],
                             OutletID=L["outlet"], AccountID=a["id"] if a is not None else "", ContactID=cid,
                             OwnerUserID=self.uid([owner])[0], LeadSource=L["source"], CreatedOn=t0,
                             StageNo=stage_no + 1, Stage=STAGES[stage_no], Status=status,
                             Probability=STAGE_PROB[stage_no], TotalUnits=total_units,
                             EstimatedRevenueINR=int(total_value), ExpectedCloseDate=exp_close,
                             ActualCloseDate=close_t, LastActivityOn=last, FinanceType=fin,
                             HasExchangeVehicle=bool(retail and r.random() < 0.3), IsTender=bool(tender),
                             LostReason=lost_reason, _owner=owner, _retail=retail, _zombie=zombie))
        self.opps, self.hist, self.lines, self.contact_rows = opps, hist, lines, contacts

    def apply_duplicate_account_split(self):
        """After a duplicate account is created, some new deals get logged against it (split history)."""
        r = self.rng
        dup_by_orig = {}
        for a in self.acc:
            if a["dup_of"] is not None:
                dup_by_orig.setdefault(self.acc[a["dup_of"]]["id"], []).append(a)
        for L in self.lead_rows:
            if L["channel"] != "Retail" and L["acct"] is not None:
                oid = self.acc[L["acct"]]["id"]
                if oid in dup_by_orig:
                    d = dup_by_orig[oid][0]
                    if L["t"] > d["created"] and r.random() < 0.35:
                        L["acct_override"] = d["id"]
        lead_map = {L["id"]: L for L in self.lead_rows}
        for o in self.opps:
            L = lead_map[o["LeadID"]]
            if L.get("acct_override"):
                o["AccountID"] = L["acct_override"]
                dup_i = self.acc_index[L["acct_override"]]
                cands = self.acc_contacts.get(dup_i, [])
                if cands:
                    o["ContactID"] = self.contact_rows[cands[0]]["ContactID"]

    # ---------------------------------------------------------------- contacts: planted issues
    def finalize_contacts(self):
        r = self.rng
        rows = self.contact_rows
        extra = []
        for c in rows:
            u = r.random()
            if c["Email"] and u < 0.10:
                c["Email"] = ""
                self.issue("contact_missing_email", "dim_contacts", c["ContactID"])
            elif c["Email"] and u < 0.13:
                e = c["Email"]
                c["Email"] = self.pick([e.replace("@", ""), e.replace(".com", ".con").replace(".in", ".inn"),
                                        e.replace("@", " @"), e.split("@")[0] + "@"])
                self.issue("contact_invalid_email", "dim_contacts", c["ContactID"], detail=c["Email"])
            v = r.random()
            if v < 0.03:
                m = c["Mobile"]
                c["Mobile"] = self.pick([m[:-1], m[0] * 10, "0" + m[:9], m[:5] + "-" + m[5:8]])
                self.issue("contact_invalid_mobile", "dim_contacts", c["ContactID"], detail=c["Mobile"])
        business = [c for c in rows if c["CustomerType"] == "Business"]
        for src in [business[i] for i in r.choice(len(business), size=int(len(business) * 0.02), replace=False)]:
            d = dict(src)
            d.update(ContactID=f"CON-{len(rows) + len(extra) + 1:06d}", FullName=self.person_variant(src["FullName"]),
                     Email="" if r.random() < 0.5 else src["Email"], CreatedOn=min(src["CreatedOn"] + r.uniform(30, 400),
                                                                                self.T - 1))
            extra.append(d)
            self.issue("duplicate_contact", "dim_contacts", d["ContactID"], src["ContactID"],
                       "Same person entered twice on the same account")
        df = pd.DataFrame(rows + extra)
        df["CreatedOn"] = self.fmt(df["CreatedOn"]).values
        self.df_contacts = df

    # ---------------------------------------------------------------- activities
    def build_activities(self):
        r, T = self.rng, self.T
        parents = []   # (type, id, owner, outlet, channel, start, end, n, unresponsive)
        for L in self.lead_rows:
            if np.isnan(L["fr"]):
                continue
            end = L["conv"] if L["status"] == "Converted" else L["last"]
            if np.isnan(end) or end < L["fr"]:
                end = L["fr"]
            n = 1 + r.poisson(3.5 if L["status"] != "Unresponsive" else 5.0)
            parents.append(("Lead", L["id"], L["owner"], L["outlet"], L["channel"], L["fr"], end, n,
                            L["status"] == "Unresponsive"))
        for o in self.opps:
            lam = (28 if o["_retail"] else 40) * (1.15 if o["Status"] == "Won" else (0.7 if o["Status"] == "Lost" else 1.0))
            if o["_zombie"]:
                lam = 5
            n = 1 + r.poisson(lam)
            parents.append(("Opportunity", o["OpportunityID"], o["_owner"], o["OutletID"], o["Channel"],
                            o["CreatedOn"], max(o["LastActivityOn"], o["CreatedOn"]), n, False))
        P = pd.DataFrame(parents, columns=["rtype", "rid", "owner", "outlet", "channel", "start", "end", "n", "unresp"])
        rep = np.repeat(np.arange(len(P)), P["n"].values)
        first_of_group = np.r_[True, rep[1:] != rep[:-1]]
        u = r.random(len(rep))
        u[first_of_group] = 1.0                                # pins each parent's last activity
        start, end = P["start"].values[rep], P["end"].values[rep]
        t = start + u * (end - start)
        day = np.floor(t)
        hour = np.clip(9 + r.random(len(rep)) * 11, 9, 20)
        t = np.where(first_of_group, t, np.maximum(day + hour / 24, start))
        t = np.minimum(t, T - 0.001)
        retail = P["channel"].values[rep] == "Retail"
        rtypes = ["Call", "WhatsApp", "Showroom Visit", "Test Drive", "Email", "Quotation Sent", "SMS Follow-up"]
        rw = np.array([35, 30, 10, 6, 7, 5, 7], float)
        btypes = ["Call", "Email", "Meeting", "WhatsApp", "Product Demo", "Site Visit", "Quotation Sent"]
        bw = np.array([30, 25, 15, 15, 5, 5, 5], float)
        act = np.where(retail, np.array(rtypes)[r.choice(7, size=len(rep), p=rw / rw.sum())],
                       np.array(btypes)[r.choice(7, size=len(rep), p=bw / bw.sum())])
        is_call = act == "Call"
        call_out = np.array(["Connected", "No Answer", "Callback Requested"])[r.choice(3, size=len(rep), p=[.55, .3, .15])]
        wa_out = np.array(["Delivered", "Replied"])[r.choice(2, size=len(rep), p=[.65, .35])]
        outcome = np.where(is_call, call_out, np.where(np.isin(act, ["WhatsApp", "SMS Follow-up", "Email"]), wa_out, "Completed"))
        unresp = P["unresp"].values[rep]
        outcome = np.where(unresp & is_call, "No Answer", outcome)
        dur_mean = {"Call": 4, "Meeting": 45, "Site Visit": 60, "Test Drive": 30, "Showroom Visit": 40, "Product Demo": 45}
        dur = np.zeros(len(rep))
        for k, m in dur_mean.items():
            mask = act == k
            dur[mask] = np.round(r.gamma(2.0, m / 2.0, mask.sum()), 1)
        dur[outcome == "No Answer"] = 0
        owners = P["owner"].values[rep]
        df = pd.DataFrame({
            "ActivityID": pd.Series(np.arange(1, len(rep) + 1)).map(lambda i: f"ACT-{i:08d}").values,
            "RegardingType": P["rtype"].values[rep], "RegardingID": P["rid"].values[rep],
            "OwnerUserID": self.uid(owners), "OutletID": P["outlet"].values[rep], "Channel": P["channel"].values[rep],
            "ActivityType": act, "Outcome": outcome, "DurationMins": dur, "_t": t})
        df = df.sort_values("_t", kind="stable").reset_index(drop=True)
        df["ActivityID"] = [f"ACT-{i:08d}" for i in range(1, len(df) + 1)]
        df["ActivityOn"] = self.fmt(df["_t"]).values
        self.df_activities = df

    # ---------------------------------------------------------------- targets & report requests
    def build_targets(self):
        r = self.rng
        ol = pd.DataFrame([o for o in self.opps if o["Status"] == "Won"])
        ol["Month"] = (self.START + pd.to_timedelta(ol["ActualCloseDate"], unit="D")).dt.to_period("M")
        avg = ol.groupby("OutletID").agg(units=("TotalUnits", "sum"), rev=("EstimatedRevenueINR", "sum"))
        n_months = self.T / 30.44
        first = self.START if self.START.day == 1 else self.START + pd.offsets.MonthBegin(1)
        months = pd.period_range(first, self.END, freq="M")      # full months only
        rows = []
        for oid in avg.index:
            mu = avg.loc[oid, "units"] / n_months
            aup = avg.loc[oid, "rev"] / max(avg.loc[oid, "units"], 1)
            for i, m in enumerate(months):
                g = 1 + 0.07 * (i / 12)
                tu = max(1, int(round(mu * SEASON[m.month] * g * r.uniform(1.0, 1.22))))
                ch = "Retail" if oid in RETAIL_OUTLETS else "Corporate & Fleet"
                rows.append(dict(MonthStart=m.start_time.strftime("%Y-%m-%d"), OutletID=oid, Channel=ch,
                                 TargetUnits=tu, TargetRevenueINR=int(round(tu * aup, -3))))
        self.df_targets = pd.DataFrame(rows)

    def build_report_requests(self):
        r, T = self.rng, self.T
        requesters = self.df_users[self.df_users["Role"].isin(
            ["GM - Sales", "Outlet Sales Manager", "Head - Corporate & Fleet Sales", "Key Account Manager"])]
        cats = {"Daily Sales Flash": ("Standing - Daily", "P1"), "Weekly Pipeline Review": ("Standing - Weekly", "P2"),
                "Target vs Achievement": ("Standing - Monthly", "P2"), "Incentive Calculation": ("Standing - Monthly", "P1"),
                "OEM Submission": ("Standing - Monthly", "P1"), "Lead Source ROI": ("Ad hoc", "P3"),
                "Data Correction": ("Ad hoc", "P2"), "Ad hoc Analysis": ("Ad hoc", "P3"),
                "Board / Owner Review Pack": ("Ad hoc", "P1")}
        cw = np.array([10, 8, 6, 4, 4, 5, 9, 12, 2], float)
        sla_h = {"P1": 4, "P2": 24, "P3": 72}
        n = int(1.9 * self.scale * T * 5 / 7)
        t = np.sort(np.floor(r.uniform(0, T, n)) + r.uniform(9.5, 18.5, n) / 24)
        rows = []
        cat_names = list(cats)
        for i, ti in enumerate(t):
            cat = cat_names[r.choice(len(cat_names), p=cw / cw.sum())]
            rtype, pri = cats[cat]
            if rtype == "Ad hoc" and r.random() < 0.2:
                pri = "P1"
            if cat == "OEM Submission":
                who = "OEM Regional Office"
            else:
                who = requesters["UserID"].values[r.integers(0, len(requesters))]
            due = ti + sla_h[pri] / 24
            hours = sla_h[pri] * r.lognormal(np.log(0.55), 0.6)
            delivered = ti + hours / 24
            status = "Delivered"
            if r.random() < 0.03:
                status, delivered = "Cancelled", np.nan
            elif delivered >= T:
                status, delivered = "In Progress", np.nan
            analyst = int(self.assign(["HQ-MIS-1", "HQ-MIS-2"], [ti], ["HQ-GM"])[0])
            rows.append(dict(RequestID=f"REQ-{i + 1:05d}", RequestedOn=ti, RequestedBy=who, Category=cat,
                             RequestType=rtype, Priority=pri, SLAHours=sla_h[pri], DueOn=due, DeliveredOn=delivered,
                             Status=status, AssignedToUserID=self.uid([analyst])[0],
                             ReworkRequired=bool(status == "Delivered" and r.random() < 0.08)))
        df = pd.DataFrame(rows)
        for c in ["RequestedOn", "DueOn", "DeliveredOn"]:
            df[c] = self.fmt(df[c]).values
        self.df_requests = df

    # ---------------------------------------------------------------- rule-based checks for the answer key
    def rule_based_issues(self):
        T = self.T
        acc_df = self.df_accounts
        for _, a in acc_df[(acc_df["KN_Designation"] == "Key") & (acc_df["DUNS"] == "")].iterrows():
            self.issue("key_account_missing_duns", "dim_accounts", a["AccountID"])
        inactive = set(self.df_users.loc[~self.df_users["IsActive"], "UserID"])
        for _, a in acc_df[acc_df["OwnerUserID"].isin(inactive)].iterrows():
            self.issue("account_owner_inactive", "dim_accounts", a["AccountID"], a["OwnerUserID"])
        for o in self.opps:
            if o["Status"] != "Open":
                continue
            if o["OwnerUserID"] in inactive:
                self.issue("open_opportunity_owner_inactive", "fact_opportunities", o["OpportunityID"], o["OwnerUserID"])
            idle = T - o["LastActivityOn"]
            if idle > 30:
                self.issue("stale_open_opportunity", "fact_opportunities", o["OpportunityID"],
                           detail=f"{int(idle)} days since last activity")
            if o["ExpectedCloseDate"] < T - 1:
                self.issue("open_opportunity_past_expected_close", "fact_opportunities", o["OpportunityID"])
        for L in self.lead_rows:
            if L["status"] in ("New", "Contacted", "Qualified") and T - L["t"] > 30:
                self.issue("unconverted_lead_over_30_days", "fact_leads", L["id"], detail=L["status"])

    # ---------------------------------------------------------------- write
    def write(self):
        os.makedirs(self.out, exist_ok=True)
        counts = {}

        def save(df, name):
            path = os.path.join(self.out, name)
            df.to_csv(path, index=False, encoding="utf-8")
            counts[name] = len(df)

        out = pd.DataFrame(OUTLETS, columns=["OutletID", "OutletName", "City", "Zone", "Channel", "_w"]).drop(columns="_w")
        out["OutletName"] = np.where(out["OutletID"] == "HQ", GROUP_NAME + " - Head Office",
                                     GROUP_NAME + " - " + out["OutletName"])
        save(out, "dim_outlets.csv")
        save(self.df_users, "dim_users.csv")
        save(self.df_products, "dim_products.csv")
        save(self.df_accounts, "dim_accounts.csv")
        save(self.df_contacts, "dim_contacts.csv")

        lr = []
        for L in self.lead_rows:
            lr.append(dict(LeadID=L["id"], CreatedOn=L["t"], Channel=L["channel"], Source=L["source"],
                           OutletID=L["outlet"], OwnerUserID=self.uid([L["owner"]])[0],
                           LeadName=L["person"] if L["channel"] == "Retail" else "",
                           CompanyName=L["company"] if L["channel"] != "Retail" else "",
                           AccountID=L.get("acct_override") or (self.acc[L["acct"]]["id"] if L["acct"] is not None else ""),
                           Mobile=L["mobile"], Email=L["email"], City=L["city"], InterestedProductID=L["prod"],
                           IsExistingCustomer=bool(L.get("repeat")), Status=L["status"],
                           FirstResponseOn=L["fr"], LastContactedOn=L["last"], ConvertedOn=L["conv"],
                           OpportunityID=L.get("opp_id", "")))
        ldf = pd.DataFrame(lr).sort_values("CreatedOn", kind="stable")
        for c in ["CreatedOn", "FirstResponseOn", "LastContactedOn", "ConvertedOn"]:
            ldf[c] = self.fmt(ldf[c]).values
        save(ldf, "fact_leads.csv")

        odf = pd.DataFrame(self.opps).drop(columns=["_owner", "_retail", "_zombie"])
        odf["CreatedOn"] = self.fmt(odf["CreatedOn"]).values
        odf["ExpectedCloseDate"] = self.fmt(odf["ExpectedCloseDate"], False).values
        odf["ActualCloseDate"] = self.fmt(odf["ActualCloseDate"], False).values
        odf["LastActivityOn"] = self.fmt(odf["LastActivityOn"]).values
        save(odf, "fact_opportunities.csv")
        save(pd.DataFrame(self.lines), "fact_opportunity_lines.csv")

        h = pd.DataFrame(self.hist, columns=["OpportunityID", "StageIdx", "EnteredOn", "ExitedOn"])
        h["StageNo"] = h["StageIdx"] + 1
        h["Stage"] = [STAGES[i] for i in h["StageIdx"]]
        h["DaysInStage"] = np.round(h["ExitedOn"] - h["EnteredOn"], 2)
        h["EnteredOn"] = self.fmt(h["EnteredOn"]).values
        h["ExitedOn"] = self.fmt(h["ExitedOn"]).values
        h.insert(0, "StageHistoryID", [f"STH-{i:07d}" for i in range(1, len(h) + 1)])
        save(h[["StageHistoryID", "OpportunityID", "StageNo", "Stage", "EnteredOn", "ExitedOn", "DaysInStage"]],
             "fact_stage_history.csv")

        act = self.df_activities
        dt = self.START + pd.to_timedelta(act["_t"], unit="D")
        fy_start = np.where(dt.dt.month >= 4, dt.dt.year, dt.dt.year - 1)
        cols = ["ActivityID", "ActivityOn", "RegardingType", "RegardingID", "OwnerUserID", "OutletID", "Channel",
                "ActivityType", "Outcome", "DurationMins"]
        os.makedirs(os.path.join(self.out, "activities"), exist_ok=True)
        for fy in sorted(set(fy_start)):
            part = act.loc[fy_start == fy, cols]
            save(part, os.path.join("activities", f"fact_activities_FY{fy}-{str(fy + 1)[-2:]}.csv"))

        save(self.df_targets, "fact_targets.csv")
        save(self.df_requests, "fact_report_requests.csv")
        key = pd.DataFrame(self.issues, columns=["IssueType", "Table", "RecordID", "RelatedRecordID", "Detail"])
        save(key, "_data_quality_answer_key.csv")

        summary = dict(dealer_group=GROUP_NAME, data_start=str(self.START.date()), data_as_of=str(self.END.date()),
                       seed=self.seed, scale=self.scale, generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                       row_counts=counts, planted_issue_counts=key["IssueType"].value_counts().to_dict(),
                       next_ids=dict(lead=len(self.lead_rows) + 1, opportunity=len(self.opps) + 1,
                                     activity=len(act) + 1, account=len(self.acc) + 1,
                                     contact=len(self.df_contacts) + 1, line=len(self.lines) + 1,
                                     request=len(self.df_requests) + 1))
        with open(os.path.join(self.out, "_generation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary

    def run(self):
        steps = [("users", self.build_users), ("products", self.build_products), ("accounts", self.build_accounts),
                 ("business contacts", self.build_b2b_contacts), ("leads", self.build_leads),
                 ("lead outcomes", self.finalize_leads), ("opportunities", self.build_opportunities),
                 ("duplicate-account history split", self.apply_duplicate_account_split),
                 ("contact data quality", self.finalize_contacts), ("activities", self.build_activities),
                 ("targets", self.build_targets), ("report requests", self.build_report_requests),
                 ("rule-based checks", self.rule_based_issues)]
        print(f"Generating {GROUP_NAME} CRM history: {self.START.date()} -> {self.END.date()} (scale {self.scale})")
        for label, fn in steps:
            t0 = time.time()
            fn()
            print(f"  {label:<34} {time.time() - t0:6.1f}s")
        summary = self.write()
        print("\nRows written:")
        for k, v in summary["row_counts"].items():
            print(f"  {k:<48} {v:>10,}")
        print(f"\nOutput folder: {os.path.abspath(self.out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic dealership CRM history")
    ap.add_argument("--scale", type=float, default=1.0, help="volume multiplier (default 1.0)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducible output")
    ap.add_argument("--years", type=float, default=3.0, help="years of history (default 3)")
    ap.add_argument("--as-of", default=None, help="last day of data, YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--out", default="data", help="output folder (default ./data)")
    Generator(ap.parse_args()).run()