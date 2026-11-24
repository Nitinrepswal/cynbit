"""
SQLite database management module for Cynbit LeadIQ.
Uses parameterized queries to safeguard against SQL injection and supports lead history persistence.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from src.config import DATABASE_PATH


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Returns a connection to SQLite with row factory enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DATABASE_PATH) -> None:
    """Initializes the SQLite schema if it does not already exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            company TEXT DEFAULT 'Unspecified',
            role TEXT DEFAULT 'Unknown',
            inquiry_text TEXT NOT NULL,
            budget_tier TEXT DEFAULT 'Unspecified / Flexible',
            timeline TEXT DEFAULT 'Exploratory / No fixed date',
            company_size TEXT DEFAULT 'Not Specified',
            source_channel TEXT DEFAULT 'Website Form',
            primary_category TEXT DEFAULT 'Other / Advisory',
            extracted_keywords TEXT DEFAULT '',
            qualification_score INTEGER DEFAULT 0,
            qualification_tier TEXT DEFAULT 'Cold',
            score_reasons TEXT DEFAULT '',
            conversion_probability REAL DEFAULT 0.0,
            ml_prediction_label TEXT DEFAULT 'Low Conversion Likelihood',
            lead_status TEXT DEFAULT 'New',
            notes TEXT DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()


def insert_lead(record: Dict[str, Any], db_path: Path = DATABASE_PATH) -> int:
    """Inserts a processed lead using parameterized SQL."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    query = """
        INSERT INTO leads (
            name, email, company, role, inquiry_text, budget_tier, timeline,
            company_size, source_channel, primary_category, extracted_keywords,
            qualification_score, qualification_tier, score_reasons,
            conversion_probability, ml_prediction_label, lead_status, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        record.get("name", "Prospective Client"),
        record.get("email", "unspecified@example.com"),
        record.get("company", "Unspecified"),
        record.get("role", "Unknown"),
        record.get("inquiry_text", ""),
        record.get("budget_tier", "Unspecified / Flexible"),
        record.get("timeline", "Exploratory / No fixed date"),
        record.get("company_size", "Not Specified"),
        record.get("source_channel", "Website Form"),
        record.get("primary_category", "Other / Advisory"),
        record.get("extracted_keywords", ""),
        int(record.get("qualification_score", 0)),
        record.get("qualification_tier", "Cold"),
        record.get("score_reasons", ""),
        float(record.get("conversion_probability", 0.0)),
        record.get("ml_prediction_label", "Low Conversion Likelihood"),
        record.get("lead_status", "New"),
        record.get("notes", "")
    )

    cursor.execute(query, params)
    lead_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return lead_id


def update_lead_status(
    lead_id: int,
    new_status: str,
    notes: Optional[str] = None,
    db_path: Path = DATABASE_PATH
) -> bool:
    """Updates lead workflow status and optional notes."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    if notes is not None:
        cursor.execute(
            "UPDATE leads SET lead_status = ?, notes = ? WHERE id = ?",
            (new_status, notes, lead_id)
        )
    else:
        cursor.execute(
            "UPDATE leads SET lead_status = ? WHERE id = ?",
            (new_status, lead_id)
        )
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return affected


def get_lead_by_id(lead_id: int, db_path: Path = DATABASE_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves single lead record by primary ID."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM leads WHERE id = ?", (lead_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_all_leads_df(
    filter_status: Optional[str] = None,
    filter_category: Optional[str] = None,
    filter_tier: Optional[str] = None,
    search_query: Optional[str] = None,
    db_path: Path = DATABASE_PATH
) -> pd.DataFrame:
    """Returns filtered leads as a pandas DataFrame."""
    init_db(db_path)
    conn = get_connection(db_path)
    query = "SELECT * FROM leads WHERE 1=1"
    params = []

    if filter_status and filter_status != "All":
        query += " AND lead_status = ?"
        params.append(filter_status)

    if filter_category and filter_category != "All":
        query += " AND primary_category = ?"
        params.append(filter_category)

    if filter_tier and filter_tier != "All":
        query += " AND qualification_tier = ?"
        params.append(filter_tier)

    if search_query:
        query += " AND (name LIKE ? OR company LIKE ? OR email LIKE ? OR inquiry_text LIKE ?)"
        term = f"%{search_query.strip()}%"
        params.extend([term, term, term, term])

    query += " ORDER BY id DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def delete_lead(lead_id: int, db_path: Path = DATABASE_PATH) -> bool:
    """Deletes a lead by primary ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM leads WHERE id = ?", (lead_id,))
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return affected


def clear_and_load_demo_leads(demo_records: List[Dict[str, Any]], db_path: Path = DATABASE_PATH) -> int:
    """Wipes leads table and loads fresh demo synthetic records."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM leads")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='leads'")
    conn.commit()
    conn.close()

    count = 0
    for rec in demo_records:
        insert_lead(rec, db_path)
        count += 1
    return count
