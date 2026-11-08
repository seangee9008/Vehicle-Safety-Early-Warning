import duckdb
import os

def clean_and_standardize():
    db_path = 'data/nhtsa.duckdb'
    
    if not os.path.exists(db_path):
        print(f"Warning: {db_path} not found. Please run this from the project root.")
        return
        
    print(f"Connecting to {db_path}...")
    conn = duckdb.connect(db_path)

    # 1. Prevent target leakage & Standardize components
    # We create a staging table that filters out 'RC' and 'EWR' and cleans the COMPDESC field
    print("Creating stg.complaints (filtering leakage and standardizing components)...")
    conn.execute("""
        CREATE SCHEMA IF NOT EXISTS stg;
        
        CREATE OR REPLACE TABLE stg.complaints AS
        SELECT 
            ODINO,
            UPPER(TRIM(MAKETXT)) AS make,
            UPPER(TRIM(MODELTXT)) AS model,
            YEARTXT AS model_year,
            -- Standardize component: extract text before the first colon
            UPPER(TRIM(SPLIT_PART(COMPDESC, ':', 1))) AS component,
            -- Convert LDATE (YYYYMMDD) to a standard Date truncated to the month
            CAST(SUBSTRING(LDATE, 1, 4) || '-' || SUBSTRING(LDATE, 5, 2) || '-01' AS DATE) AS complaint_month,
            CDESCR as narrative,
            -- Severity flags
            CASE WHEN CRASH = 'Y' THEN 1 ELSE 0 END AS is_crash,
            CASE WHEN FIRE = 'Y' THEN 1 ELSE 0 END AS is_fire,
            TRY_CAST(INJURED AS INTEGER) AS injured_count,
            TRY_CAST(DEATHS AS INTEGER) AS deaths_count
        FROM raw.complaints
        -- Prevent Target Leakage: Drop recall complaints and early warning reports
        WHERE CMPL_TYPE NOT IN ('RC', 'EWR')
          AND LDATE IS NOT NULL 
          AND LDATE != ''
          AND LENGTH(CAST(LDATE AS VARCHAR)) = 8
    """)

    # 2. Align the Prediction Unit & Build Feature Foundation
    # Final modeling table aggregated to one row per make + model + model year + component per month
    print("Creating int.monthly_features (aggregating prediction unit & monthly rollups)...")
    conn.execute("""
        CREATE SCHEMA IF NOT EXISTS int;
        
        CREATE OR REPLACE TABLE int.monthly_features AS
        SELECT 
            make,
            model,
            model_year,
            component,
            complaint_month,
            COUNT(*) AS monthly_complaints,
            SUM(is_crash) AS monthly_crashes,
            SUM(is_fire) AS monthly_fires,
            SUM(COALESCE(injured_count, 0)) AS monthly_injured,
            SUM(COALESCE(deaths_count, 0)) AS monthly_deaths
        FROM stg.complaints
        GROUP BY 1, 2, 3, 4, 5
    """)

    print("\nVerification Summary:")
    stg_count = conn.sql("SELECT COUNT(*) FROM stg.complaints").fetchall()[0][0]
    int_count = conn.sql("SELECT COUNT(*) FROM int.monthly_features").fetchall()[0][0]
    print(f"-> stg.complaints rows (after leakage filter): {stg_count:,}")
    print(f"-> int.monthly_features rows (aggregated prediction units): {int_count:,}")
    
    print("\nSample of Prediction Unit Data (int.monthly_features):")
    print(conn.sql("SELECT * FROM int.monthly_features LIMIT 5").df().to_string(index=False))

    conn.close()
    print("\nData cleaning and standardization complete. Ready for modeling")

if __name__ == "__main__":
    clean_and_standardize()
