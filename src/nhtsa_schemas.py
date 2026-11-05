"""Column names for NHTSA ODI flat files (tab-delimited, no header row).

Official dictionaries (downloaded to data/raw/nhtsa/dictionaries/ by download_nhtsa.py):
  Complaints:     https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt
  Recalls:        https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt
  Investigations: https://static.nhtsa.gov/odi/ffdd/inv/INV.txt

COMPLAINTS and INVESTIGATIONS match the dictionaries (CMPL.txt dated April 30, 2026).
RECALLS matches the 29 fields in RCL.txt (May 2025).
"""

COMPLAINTS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH",
    "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN",
    "DATEA", "LDATE", "MILES", "OCCURENCES", "CDESCR", "CMPL_TYPE",
    "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN",
    "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE",
    "TRANS_TYPE", "VEH_SPEED", "DOT", "TIRE_SIZE", "LOC_OF_TIRE",
    "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE",
    "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY",
    "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN",
    "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
]

INVESTIGATIONS = [
    "ACTION_NUMBER", "MAKE", "MODEL", "YEAR", "COMPNAME", "MFR_NAME",
    "ODATE", "CDATE", "CAMPNO", "SUBJECT", "SUMMARY",
]

RECALLS = [
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO",
    "COMPNAME", "MFGNAME", "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE",
    "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA", "RPNO", "FMVSS",
    "DESC_DEFECT", "CONSEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES",
    "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO",
    "DO_NOT_DRIVE", "PARK_OUTSIDE",
]

TABLES = {"complaints": COMPLAINTS, "recalls": RECALLS, "investigations": INVESTIGATIONS}

# Personal fields nobody on the team needs; dropped at load time.
DROP = {"complaints": ["VIN", "CITY", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY",
                       "DEALER_STATE", "DEALER_ZIP", "VEHICLE_OPERATOR"]}
