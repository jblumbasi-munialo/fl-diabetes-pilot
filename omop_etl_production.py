#!/usr/bin/env python3
"""
Production-grade OMOP ETL script for diabetes (PIMA) data.
Maps source CSV to OMOP CDM tables: person, measurement, condition_occurrence.
Features error handling, logging, configuration, data validation, and idempotency.
"""

import os
import sys
import json
import logging
import hashlib
import sqlite3
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from contextlib import contextmanager

# ==========================
# 1. Configuration
# ==========================
CONFIG_FILE = "omop_etl_config.json"

DEFAULT_CONFIG = {
    "source": {
        "csv_path": "data/pima_node_A.csv",   # or node_B
        "has_header": False,
        "columns": ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                    "insulin", "bmi", "pedigree", "age", "target"]
    },
    "target_db": {
        "type": "sqlite",                     # "sqlite" or "postgres"
        "sqlite_path": "omop_diabetes.db",
        "postgres": {
            "host": "localhost",
            "port": 5432,
            "database": "omop",
            "user": "postgres",
            "password": ""
        }
    },
    "etl": {
        "batch_size": 1000,
        "drop_existing_tables": False,
        "pseudonymization_salt": "diabetes_fl_salt_2025"
    },
    "concept_mappings": {
        "glucose": {"concept_id": 1554, "unit_concept_id": 8840},
        "blood_pressure": {"concept_id": 8480, "unit_concept_id": 8876},
        "bmi": {"concept_id": 3049187, "unit_concept_id": 9559},
        "insulin": {"concept_id": 3013762, "unit_concept_id": 8534},
        "skin_thickness": {"concept_id": 3016261, "unit_concept_id": 8576},
        "pregnancies": {"concept_id": 4085704, "unit_concept_id": None},
        "pedigree": {"concept_id": 4015197, "unit_concept_id": None}
    },
    "condition_concepts": {
        "diabetes_mellitus_type_2": 44054006,   # SNOMED
        "condition_status_present": 32908
    }
}

# ==========================
# 2. Logging setup
# ==========================
def setup_logging(log_file="omop_etl.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ==========================
# 3. Custom exceptions
# ==========================
class ETLValidationError(Exception):
    pass

class ETLDatabaseError(Exception):
    pass

# ==========================
# 4. Database connection manager
# ==========================
@contextmanager
def get_db_connection(config: Dict[str, Any]):
    """Context manager for database connection (SQLite or PostgreSQL)."""
    db_type = config["target_db"]["type"]
    if db_type == "sqlite":
        db_path = config["target_db"]["sqlite_path"]
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    elif db_type == "postgres":
        pg = config["target_db"]["postgres"]
        conn = psycopg2.connect(
            host=pg["host"],
            port=pg["port"],
            database=pg["database"],
            user=pg["user"],
            password=pg["password"]
        )
        conn.autocommit = False
        try:
            yield conn
        finally:
            conn.close()
    else:
        raise ETLDatabaseError(f"Unsupported database type: {db_type}")

# ==========================
# 5. Table creation (OMOP schema subset)
# ==========================
def create_omop_tables(conn, drop_existing: bool = False):
    """Create OMOP CDM tables (person, measurement, condition_occurrence)."""
    cursor = conn.cursor()
    if drop_existing:
        cursor.execute("DROP TABLE IF EXISTS condition_occurrence")
        cursor.execute("DROP TABLE IF EXISTS measurement")
        cursor.execute("DROP TABLE IF EXISTS person")
        logger.info("Dropped existing OMOP tables")

    # Person table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS person (
            person_id INTEGER PRIMARY KEY,
            person_source_value TEXT UNIQUE,
            year_of_birth INTEGER,
            gender_concept_id INTEGER,
            race_concept_id INTEGER,
            ethnicity_concept_id INTEGER
        )
    """)

    # Measurement table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS measurement (
            measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            measurement_concept_id INTEGER,
            measurement_date DATE,
            value_as_number REAL,
            unit_concept_id INTEGER,
            FOREIGN KEY (person_id) REFERENCES person(person_id)
        )
    """)

    # Condition occurrence table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS condition_occurrence (
            condition_occurrence_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            condition_concept_id INTEGER,
            condition_start_date DATE,
            condition_status_concept_id INTEGER,
            FOREIGN KEY (person_id) REFERENCES person(person_id)
        )
    """)

    conn.commit()
    logger.info("OMOP tables created/verified")

# ==========================
# 6. Data validation
# ==========================
def validate_source_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Validate and clean source data: range checks, missing values."""
    expected_cols = config["source"]["columns"]
    if list(df.columns) != expected_cols:
        raise ETLValidationError(f"Column mismatch. Expected {expected_cols}, got {list(df.columns)}")

    # Range checks (clinical plausible values)
    validations = {
        "glucose": (0, 500),
        "blood_pressure": (0, 250),
        "bmi": (10, 80),
        "age": (0, 120),
        "pregnancies": (0, 20),
        "insulin": (0, 2000),
        "skin_thickness": (0, 100),
        "pedigree": (0, 3),
        "target": (0, 1)
    }
    for col, (min_val, max_val) in validations.items():
        if col in df.columns:
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            if out_of_range.any():
                logger.warning(f"{col}: {out_of_range.sum()} values out of range [{min_val},{max_val}]")
                df.loc[out_of_range, col] = np.nan

    # Drop rows with missing target or essential features (glucose, bmi)
    essential = ["glucose", "bmi", "target"]
    before = len(df)
    df = df.dropna(subset=essential)
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with missing essential features")

    return df

# ==========================
# 7. Helper: pseudonymisation
# ==========================
def pseudonymise(row_idx: int, salt: str) -> str:
    """Create irreversible pseudonym for person_source_value."""
    raw = f"SOURCE_{row_idx}"
    return hashlib.sha256((raw + salt).encode()).hexdigest()

# ==========================
# 8. ETL core logic
# ==========================
def run_etl(config: Dict[str, Any]):
    """Main ETL pipeline."""
    logger.info("Starting OMOP ETL")
    csv_path = config["source"]["csv_path"]
    if not os.path.exists(csv_path):
        raise ETLValidationError(f"Source CSV not found: {csv_path}")

    # ----- Extract -----
    logger.info(f"Extracting data from {csv_path}")
    df = pd.read_csv(csv_path, header=None, names=config["source"]["columns"])
    logger.info(f"Loaded {len(df)} rows")

    # ----- Transform & Validate -----
    df = validate_source_data(df, config)
    logger.info(f"After validation: {len(df)} rows")

    # Add a row index for pseudonymisation
    df.reset_index(drop=True, inplace=True)
    df["row_idx"] = df.index

    # ----- Load into OMOP -----
    with get_db_connection(config) as conn:
        create_omop_tables(conn, drop_existing=config["etl"]["drop_existing_tables"])
        cursor = conn.cursor()

        # Check existing persons to avoid duplicates
        cursor.execute("SELECT person_source_value FROM person")
        existing_sources = set(row[0] for row in cursor.fetchall())

        inserted_persons = 0
        inserted_measurements = 0
        inserted_conditions = 0

        # Process in batches
        batch_size = config["etl"]["batch_size"]
        salt = config["etl"]["pseudonymization_salt"]
        current_year = datetime.now().year

        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start+batch_size]
            for _, row in batch.iterrows():
                # Person
                person_source = pseudonymise(row["row_idx"], salt)
                if person_source in existing_sources:
                    # Get existing person_id
                    cursor.execute("SELECT person_id FROM person WHERE person_source_value = ?", (person_source,))
                    person_id = cursor.fetchone()[0]
                else:
                    birth_year = current_year - int(row["age"]) if pd.notna(row["age"]) else None
                    cursor.execute("""
                        INSERT INTO person (person_source_value, year_of_birth, gender_concept_id)
                        VALUES (?, ?, ?)
                    """, (person_source, birth_year, 0))  # 0 = unknown gender
                    person_id = cursor.lastrowid
                    existing_sources.add(person_source)
                    inserted_persons += 1

                # Measurements
                measurement_date = datetime.now().date().isoformat()
                for field, mapping in config["concept_mappings"].items():
                    if field in row and pd.notna(row[field]) and row[field] > 0:
                        cursor.execute("""
                            INSERT INTO measurement (person_id, measurement_concept_id, measurement_date, value_as_number, unit_concept_id)
                            VALUES (?, ?, ?, ?, ?)
                        """, (person_id, mapping["concept_id"], measurement_date, float(row[field]), mapping["unit_concept_id"]))
                        inserted_measurements += 1

                # Condition (diabetes diagnosis)
                if row["target"] == 1:
                    cond_concept = config["condition_concepts"]["diabetes_mellitus_type_2"]
                    status_concept = config["condition_concepts"]["condition_status_present"]
                    cursor.execute("""
                        INSERT INTO condition_occurrence (person_id, condition_concept_id, condition_start_date, condition_status_concept_id)
                        VALUES (?, ?, ?, ?)
                    """, (person_id, cond_concept, measurement_date, status_concept))
                    inserted_conditions += 1

            conn.commit()
            logger.info(f"Batch {start//batch_size + 1} committed. Persons: {inserted_persons}, Measurements: {inserted_measurements}, Conditions: {inserted_conditions}")

        logger.info(f"ETL completed. Total inserted: {inserted_persons} persons, {inserted_measurements} measurements, {inserted_conditions} conditions")

# ==========================
# 9. Entry point
# ==========================
def main():
    # Load or create configuration
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {CONFIG_FILE}")
    else:
        config = DEFAULT_CONFIG
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Created default configuration file {CONFIG_FILE}. Please edit if needed.")
        print(f"Please review {CONFIG_FILE} and run again.")
        sys.exit(0)

    try:
        run_etl(config)
    except (ETLValidationError, ETLDatabaseError, Exception) as e:
        logger.exception(f"ETL failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
