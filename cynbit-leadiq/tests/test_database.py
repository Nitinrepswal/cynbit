"""
Unit and integration tests for SQLite persistence in Cynbit LeadIQ.
"""

import tempfile
from pathlib import Path
from src.database import (
    init_db,
    insert_lead,
    get_lead_by_id,
    get_all_leads_df,
    update_lead_status,
    delete_lead,
    clear_and_load_demo_leads
)


def test_database_crud_operations():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        init_db(db_path)

        record = {
            "name": "Sarah Connor",
            "email": "sarah@cyberdyne.org",
            "company": "Cyberdyne Systems",
            "role": "Director of AI",
            "inquiry_text": "Need neural net safety and security audit on AWS Kubernetes.",
            "budget_tier": "> $50,000 (Enterprise)",
            "timeline": "Immediate (< 2 weeks)",
            "company_size": "Enterprise (500+ employees)",
            "source_channel": "Direct Inbound",
            "primary_category": "AI/ML Solutions",
            "extracted_keywords": "neural net (1.2), security (0.9)",
            "qualification_score": 88,
            "qualification_tier": "Hot",
            "score_reasons": "+20 pts [Clarity]",
            "conversion_probability": 0.89,
            "ml_prediction_label": "High Conversion Likelihood",
            "lead_status": "New",
            "notes": "Urgent review"
        }

        # 1. Insert
        lead_id = insert_lead(record, db_path=db_path)
        assert lead_id is not None
        assert lead_id > 0

        # 2. Read
        fetched = get_lead_by_id(lead_id, db_path=db_path)
        assert fetched is not None
        assert fetched["name"] == "Sarah Connor"
        assert fetched["qualification_score"] == 88

        # 3. Update Status
        updated = update_lead_status(lead_id, "Qualified", notes="Passed initial review", db_path=db_path)
        assert updated is True
        fetched_updated = get_lead_by_id(lead_id, db_path=db_path)
        assert fetched_updated["lead_status"] == "Qualified"
        assert fetched_updated["notes"] == "Passed initial review"

        # 4. Filter DF
        df = get_all_leads_df(filter_status="Qualified", db_path=db_path)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Sarah Connor"

        # 5. Delete
        deleted = delete_lead(lead_id, db_path=db_path)
        assert deleted is True
        assert get_lead_by_id(lead_id, db_path=db_path) is None

    finally:
        if db_path.exists():
            db_path.unlink()


def test_clear_and_load_demo_leads():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        sample_demo = [
            {
                "name": f"Demo User {i}",
                "email": f"demo{i}@test.com",
                "inquiry_text": "Sample inquiry text for testing reset functionality.",
                "qualification_score": 75,
                "qualification_tier": "Hot",
                "conversion_probability": 0.82
            }
            for i in range(5)
        ]

        count = clear_and_load_demo_leads(sample_demo, db_path=db_path)
        assert count == 5
        df = get_all_leads_df(db_path=db_path)
        assert len(df) == 5

    finally:
        if db_path.exists():
            db_path.unlink()
