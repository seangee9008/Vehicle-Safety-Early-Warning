"""
One-time database cleaning and component standardization script for NHTSA data.
Creates/updates `stg.complaints`, `stg.recalls`, and `int.monthly_features` in `data/nhtsa.duckdb`.
Standardizes vehicle makes, models, and top-level component names across both tables.

Usage (from project root):
    python scripts/nhtsa_data_cleaning_v2.py
"""

import os
import duckdb


# SQL macro logic to normalize top-level component names across complaints (COMPDESC) and recalls (COMPNAME).
# 1. Takes the text before the first colon or slash (e.g. "AIR BAGS:FRONTAL" -> "AIR BAGS")
# 2.Collapses extra whitespace and converts to uppercase
# 3. Maps synonymous/inconsistent NHTSA component labels into uniform canonical categories
COMPONENT_STANDARDIZATION_SQL = """
    CREATE OR REPLACE MACRO standardize_component(raw_comp) AS (
        WITH extracted AS (
            SELECT REGEXP_REPLACE(
                UPPER(TRIM(SPLIT_PART(SPLIT_PART(COALESCE(CAST(raw_comp AS VARCHAR), ''), ':', 1), '/', 1))),
                '\s+', ' ', 'g'
            ) AS base_comp
        )
        SELECT CASE
            WHEN base_comp = '' OR base_comp IS NULL THEN 'UNKNOWN OR OTHER'
            WHEN base_comp LIKE 'AIR BAG%' OR base_comp LIKE 'AIRBAG%' THEN 'AIR BAGS'
            WHEN base_comp LIKE 'SERVICE BRAKE%' OR base_comp LIKE 'BRAKE%' OR base_comp LIKE 'PARKING BRAKE%' THEN 'BRAKES'
            WHEN base_comp LIKE 'ENGINE%' THEN 'ENGINE AND ENGINE COOLING'
            WHEN base_comp LIKE 'FUEL%' THEN 'FUEL/PROPULSION SYSTEM'
            WHEN base_comp LIKE 'ELECTRICAL%' THEN 'ELECTRICAL SYSTEM'
            WHEN base_comp LIKE 'STEERING%' THEN 'STEERING'
            WHEN base_comp LIKE 'SUSPENSION%' THEN 'SUSPENSION'
            WHEN base_comp LIKE 'POWER TRAIN%' OR base_comp LIKE 'POWERTRAIN%' THEN 'POWER TRAIN'
            WHEN base_comp LIKE 'STRUCTURE%' OR base_comp LIKE 'BODY%' THEN 'STRUCTURE'
            WHEN base_comp LIKE 'SEAT BELT%' OR base_comp LIKE 'SEATBELT%' THEN 'SEAT BELTS'
            WHEN base_comp LIKE 'SEAT%' THEN 'SEATS'
            WHEN base_comp LIKE 'TIRE%' THEN 'TIRES'
            WHEN base_comp LIKE 'WHEEL%' THEN 'WHEELS'
            WHEN base_comp LIKE 'VISIBILITY%' OR base_comp LIKE 'WINDSHIELD%' OR base_comp LIKE 'WIPER%' THEN 'VISIBILITY'
            WHEN base_comp LIKE 'EXTERIOR LIGHT%' OR base_comp LIKE 'INTERIOR LIGHT%' OR base_comp LIKE 'LIGHT%' THEN 'LIGHTING'
            WHEN base_comp LIKE 'LATCH%' OR base_comp LIKE 'DOOR%' OR base_comp LIKE 'LOCKS%' THEN 'LATCHES/LOCKS/LINKAGES'
            WHEN base_comp LIKE 'VEHICLE SPEED%' OR base_comp LIKE 'CRUISE%' OR base_comp LIKE 'THROTTLE%' THEN 'VEHICLE SPEED CONTROL'
            WHEN base_comp LIKE 'FORWARD COLLISION%' OR base_comp LIKE 'LANE DEPARTURE%' OR base_comp LIKE 'BACK OVER%' OR base_comp LIKE 'ADAS%' OR base_comp LIKE 'ELECTRONIC STABILITY%' THEN 'ADAS AND STABILITY CONTROL'
            WHEN base_comp LIKE 'CHILD SEAT%' OR base_comp LIKE 'EQUIPMENT%' OR base_comp LIKE 'TRAILER%' THEN 'EQUIPMENT AND ACCESSORIES'
            WHEN base_comp LIKE 'UNKNOWN%' OR base_comp LIKE 'OTHER%' THEN 'UNKNOWN OR OTHER'
            ELSE base_comp
        END
        FROM extracted
    );
"""

# SQL macro to standardize vehicle Make names across complaints and recalls
MAKE_STANDARDIZATION_SQL = """
    CREATE OR REPLACE MACRO standardize_make(raw_make) AS (
        WITH cleaned AS (
            SELECT REGEXP_REPLACE(UPPER(TRIM(COALESCE(CAST(raw_make AS VARCHAR), ''))), '[^A-Z0-9 ]', '', 'g') AS m
        )
        SELECT CASE
            WHEN m IN ('CHEVY', 'CHEVROLET', 'CHEV', 'CHEVROLET MOTOR DIV') THEN 'CHEVROLET'
            WHEN m IN ('MERCEDES', 'MERCEDES BENZ', 'MERCEDESBENZ', 'DAIMLER BENZ') THEN 'MERCEDES-BENZ'
            WHEN m IN ('VW', 'VOLKSWAGEN', 'VOLKSWAGON') THEN 'VOLKSWAGEN'
            WHEN m IN ('DODGE', 'DODGE RAM') THEN 'DODGE'
            WHEN m IN ('FORD', 'FORD MOTOR COMPANY') THEN 'FORD'
            WHEN m IN ('GMC', 'GENERAL MOTORS', 'GM') THEN 'GMC'
            WHEN m IN ('TOYOTA', 'TOYOTA MOTOR SALES') THEN 'TOYOTA'
            WHEN m IN ('HONDA', 'AMERICAN HONDA') THEN 'HONDA'
            WHEN m IN ('NISSAN', 'NISSAN NORTH AMERICA') THEN 'NISSAN'
            WHEN m IN ('HYUNDAI', 'HYUNDAI MOTOR AMERICA') THEN 'HYUNDAI'
            WHEN m IN ('KIA', 'KIA MOTORS', 'KIA AMERICA') THEN 'KIA'
            WHEN m IN ('FCA', 'CHRYSLER') THEN 'CHRYSLER'
            ELSE m
        END
        FROM cleaned
    );
"""

# SQL macro to normalize Model names (strip punctuation/hyphens/extra spaces so e.g. "F-150" and "F 150" -> "F150")
MODEL_STANDARDIZATION_SQL = """
    CREATE OR REPLACE MACRO standardize_model(raw_model) AS (
        REGEXP_REPLACE(
            UPPER(TRIM(COALESCE(CAST(raw_model AS VARCHAR), ''))),
            '[\s\-_./]+', '', 'g'
        )
    );
"""


def clean_database():
    db_path = "data/nhtsa.duckdb"

    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found. Please run this script from the project root directory.")
        return

    print(f"Connecting to {db_path}...")
    conn = duckdb.connect(db_path)

    try:
        # Register standardization macros in DuckDB
        print("Registering standardization SQL macros for Make, Model, and Component...")
        conn.execute(COMPONENT_STANDARDIZATION_SQL)
        conn.execute(MAKE_STANDARDIZATION_SQL)
        conn.execute(MODEL_STANDARDIZATION_SQL)

        conn.execute("CREATE SCHEMA IF NOT EXISTS stg;")
        conn.execute("CREATE SCHEMA IF NOT EXISTS int;")

        # -------------------------------------------------------------------------
        # 1. Clean & Standardize stg.complaints
        # -------------------------------------------------------------------------
        print("Building stg.complaints...")
        conn.execute("""
            CREATE OR REPLACE TABLE stg.complaints AS
            SELECT
                CMPLID AS cmpl_id,
                ODINO AS odi_no,
                standardize_make(MAKETXT) AS make,
                standardize_model(MODELTXT) AS model,
                UPPER(TRIM(MODELTXT)) AS model_raw,
                TRY_CAST(YEARTXT AS INTEGER) AS model_year,
                standardize_component(COMPDESC) AS component,
                UPPER(TRIM(COMPDESC)) AS component_raw,
                TRY_STRPTIME(CAST(LDATE AS VARCHAR), '%Y%m%d')::DATE AS ldate,
                DATE_TRUNC('month', TRY_STRPTIME(CAST(LDATE AS VARCHAR), '%Y%m%d')::DATE) AS complaint_month,
                TRY_STRPTIME(CAST(FAILDATE AS VARCHAR), '%Y%m%d')::DATE AS fail_date,
                CASE WHEN UPPER(TRIM(CRASH)) = 'Y' THEN 1 ELSE 0 END AS is_crash,
                CASE WHEN UPPER(TRIM(FIRE)) = 'Y' THEN 1 ELSE 0 END AS is_fire,
                COALESCE(TRY_CAST(INJURED AS INTEGER), 0) AS injured_count,
                COALESCE(TRY_CAST(DEATHS AS INTEGER), 0) AS deaths_count,
                TRY_CAST(MILES AS DOUBLE) AS miles,
                CDESCR AS narrative,
                UPPER(TRIM(CMPL_TYPE)) AS cmpl_type,
                UPPER(TRIM(PROD_TYPE)) AS prod_type
            FROM raw.complaints
            WHERE COALESCE(UPPER(TRIM(CMPL_TYPE)), '') NOT IN ('RC', 'EWR')
              AND COALESCE(UPPER(TRIM(PROD_TYPE)), 'V') = 'V'
              AND TRY_CAST(YEARTXT AS INTEGER) BETWEEN 1980 AND 2027
              AND TRY_STRPTIME(CAST(LDATE AS VARCHAR), '%Y%m%d') IS NOT NULL
              AND MAKETXT IS NOT NULL AND TRIM(MAKETXT) != ''
              AND MODELTXT IS NOT NULL AND TRIM(MODELTXT) != ''
        """)

        # -------------------------------------------------------------------------
        # 2. Clean & Standardize stg.recalls
        # -------------------------------------------------------------------------
        print("Building stg.recalls...")
        conn.execute("""
            CREATE OR REPLACE TABLE stg.recalls AS
            SELECT
                RECORD_ID AS record_id,
                UPPER(TRIM(CAMPNO)) AS camp_no,
                standardize_make(MAKETXT) AS make,
                standardize_model(MODELTXT) AS model,
                UPPER(TRIM(MODELTXT)) AS model_raw,
                TRY_CAST(YEARTXT AS INTEGER) AS model_year,
                standardize_component(COMPNAME) AS component,
                UPPER(TRIM(COMPNAME)) AS component_raw,
                TRY_STRPTIME(CAST(RCDATE AS VARCHAR), '%Y%m%d')::DATE AS recall_date,
                DATE_TRUNC('month', TRY_STRPTIME(CAST(RCDATE AS VARCHAR), '%Y%m%d')::DATE) AS recall_month,
                TRY_STRPTIME(CAST(ODATE AS VARCHAR), '%Y%m%d')::DATE AS owner_notif_date,
                COALESCE(TRY_CAST(POTAFF AS INTEGER), 0) AS pot_affected,
                UPPER(TRIM(RCLTYPECD)) AS recall_type,
                DESC_DEFECT AS defect_summary,
                CONSEQUENCE_DEFECT AS consequence_summary,
                CORRECTIVE_ACTION AS corrective_action
            FROM raw.recalls
            WHERE COALESCE(UPPER(TRIM(RCLTYPECD)), 'V') = 'V'
              AND TRY_CAST(YEARTXT AS INTEGER) BETWEEN 1980 AND 2027
              AND TRY_STRPTIME(CAST(RCDATE AS VARCHAR), '%Y%m%d') IS NOT NULL
              AND MAKETXT IS NOT NULL AND TRIM(MAKETXT) != ''
              AND MODELTXT IS NOT NULL AND TRIM(MODELTXT) != ''
        """)

        # -------------------------------------------------------------------------
        # 3. Build int.monthly_features (Contract Table for Modeling & Agent)
        # -------------------------------------------------------------------------
        print("Building int.monthly_features (aggregating to make + model + model_year + component + month)...")
        conn.execute("""
            CREATE OR REPLACE TABLE int.monthly_features AS
            SELECT
                make,
                model,
                model_year,
                component,
                complaint_month,
                COUNT(*) AS monthly_complaints,
                COUNT(DISTINCT odi_no) AS unique_incident_complaints,
                SUM(is_crash) AS monthly_crashes,
                SUM(is_fire) AS monthly_fires,
                SUM(injured_count) AS monthly_injured,
                SUM(deaths_count) AS monthly_deaths,
                AVG(miles) AS avg_miles_at_failure
            FROM stg.complaints
            WHERE component != 'UNKNOWN OR OTHER'
            GROUP BY 1, 2, 3, 4, 5
        """)

        # -------------------------------------------------------------------------
        # 4. Verification & Diagnostics Output
        # -------------------------------------------------------------------------
        print("\n" + "=" * 65)
        print(" VERIFICATION SUMMARY")
        print("=" * 65)

        cmpl_count = conn.sql("SELECT COUNT(*) FROM stg.complaints").fetchall()[0][0]
        rcl_count = conn.sql("SELECT COUNT(*) FROM stg.recalls").fetchall()[0][0]
        feat_count = conn.sql("SELECT COUNT(*) FROM int.monthly_features").fetchall()[0][0]

        print(f"stg.complaints rows       : {cmpl_count:,}")
        print(f"stg.recalls rows          : {rcl_count:,}")
        print(f"int.monthly_features rows : {feat_count:,}")

        print("\nTop Standardized Components Across Both Tables:")
        comp_check = conn.sql("""
            WITH c AS (
                SELECT component, COUNT(*) AS cmpl_cnt FROM stg.complaints GROUP BY 1
            ),
            r AS (
                SELECT component, COUNT(*) AS rcl_cnt FROM stg.recalls GROUP BY 1
            )
            SELECT
                COALESCE(c.component, r.component) AS standardized_component,
                COALESCE(c.cmpl_cnt, 0) AS complaints_count,
                COALESCE(r.rcl_cnt, 0) AS recalls_count
            FROM c
            FULL OUTER JOIN r ON c.component = r.component
            ORDER BY complaints_count DESC
            LIMIT 15
        """).df()
        print(comp_check.to_string(index=False))

        # Check how many recall vehicle-components match complaint vehicle-components prior to recall_date
        match_check = conn.sql("""
            WITH distinct_recalls AS (
                SELECT make, model, model_year, component, MIN(recall_date) AS first_recall_date
                FROM stg.recalls
                WHERE component != 'UNKNOWN OR OTHER'
                  AND recall_date >= DATE '2015-01-01'
                GROUP BY 1, 2, 3, 4
            ),
            matched AS (
                SELECT
                    r.make, r.model, r.model_year, r.component,
                    COUNT(c.odi_no) AS pre_recall_complaints
                FROM distinct_recalls r
                LEFT JOIN stg.complaints c
                  ON r.make = c.make
                 AND r.model = c.model
                 AND r.model_year = c.model_year
                 AND r.component = c.component
                 AND c.ldate < r.first_recall_date
                GROUP BY 1, 2, 3, 4
            )
            SELECT
                COUNT(*) AS total_recalled_units_since_2015,
                SUM(CASE WHEN pre_recall_complaints > 0 THEN 1 ELSE 0 END) AS units_with_prior_complaints,
                ROUND(100.0 * SUM(CASE WHEN pre_recall_complaints > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS match_rate_pct
            FROM matched
        """).df()
        print("\nJoin Verification (Recalls Since 2015 Matched to Pre-Recall Complaints):")
        print(match_check.to_string(index=False))

    finally:
        conn.close()
        print("\nDatabase connection closed cleanly. One-time cleaning complete!")


if __name__ == "__main__":
    clean_database()
