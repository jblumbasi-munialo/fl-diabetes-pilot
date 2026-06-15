"""
OMOP Data Loader for Fed‑BioMed.
Reads person, measurement, and condition tables and reconstructs the feature matrix + target.
Assumes the OMOP database was populated by omop_etl_production.py.
"""

import sqlite3
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Mapping from measurement concept IDs to feature names (as in source CSV)
CONCEPT_TO_FEATURE = {
    1554: "glucose",        # LOINC glucose
    8480: "blood_pressure", # LOINC systolic? we use diastolic? Adapt if needed
    3049187: "bmi",
    3013762: "insulin",
    3016261: "skin_thickness",
    4085704: "pregnancies",
    4015197: "pedigree",
    # Note: age comes from person.year_of_birth, not a measurement.
}

def load_omop_data(db_path: str, table_prefix: str = "") -> pd.DataFrame:
    """
    Load OMOP data and pivot into a feature matrix with binary target.

    Args:
        db_path: Path to SQLite database (or PostgreSQL connection string)
        table_prefix: Optional prefix for tables (unused here)

    Returns:
        DataFrame with columns: pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, pedigree, age, target
    """
    conn = sqlite3.connect(db_path)
    # 1. Load persons (age computed from year_of_birth)
    persons = pd.read_sql_query("""
        SELECT person_id, year_of_birth,
               CAST(strftime('%Y', 'now') - year_of_birth AS INTEGER) AS age
        FROM person
    """, conn)
    persons = persons[persons["age"].notna() & (persons["age"] >= 0)]

    # 2. Load measurements (pivot: one row per person)
    measurements = pd.read_sql_query("""
        SELECT person_id, measurement_concept_id, value_as_number
        FROM measurement
        WHERE measurement_concept_id IN ({})
    """.format(','.join(map(str, CONCEPT_TO_FEATURE.keys()))), conn)

    # Pivot measurements to wide format
    meas_pivot = measurements.pivot(index="person_id", columns="measurement_concept_id", values="value_as_number")
    meas_pivot.columns = [CONCEPT_TO_FEATURE.get(col, col) for col in meas_pivot.columns]

    # 3. Load conditions (target = 1 if diabetes diagnosis present)
    conditions = pd.read_sql_query("""
        SELECT DISTINCT person_id, 1 AS target
        FROM condition_occurrence
        WHERE condition_concept_id = 44054006   -- type 2 diabetes
    """, conn)

    # 4. Merge all
    df = persons.merge(meas_pivot, on="person_id", how="left")
    df = df.merge(conditions, on="person_id", how="left")
    df["target"] = df["target"].fillna(0).astype(int)

    # 5. Ensure all required columns exist (fill missing with NaN)
    required = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                "insulin", "bmi", "pedigree", "age", "target"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    # 6. Drop rows with missing target or essential features
    df = df.dropna(subset=["glucose", "bmi", "target"])
    # Also drop rows with age missing (should not happen)
    df = df.dropna(subset=["age"])

    conn.close()
    logger.info(f"Loaded {len(df)} patients from OMOP database")
    return df[required].copy()
