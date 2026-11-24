"""
Unit tests for What-If scenario simulator.
"""

from src.prediction import ConversionPredictor
from src.simulator import simulate_lead_modifications


def test_simulate_lead_modifications_calculates_positive_delta():
    predictor = ConversionPredictor()
    assert predictor.is_ready()

    base_lead = {
        "name": "Test Client",
        "email": "test@startup.io",
        "inquiry_text": "Need some web development work.",
        "budget_tier": "< $5,000 (Low)",
        "timeline": "Exploratory / No fixed date",
        "company_size": "Solo Founder / Individual",
        "role": "Founder",
        "source_channel": "Website Form",
        "primary_category": "Web Development"
    }

    simulated = simulate_lead_modifications(
        predictor=predictor,
        base_lead=base_lead,
        new_inquiry_text=(
            "We have secured executive budget to build an enterprise web portal with React, Next.js, "
            "and Python FastAPI on AWS. Full wireframes and user stories ready. Target launch within 4 weeks."
        ),
        new_budget="> $50,000 (Enterprise)",
        new_timeline="Immediate (< 2 weeks)",
        new_company_size="Enterprise (500+ employees)",
        new_role="VP of Engineering"
    )

    assert simulated["deltas"]["score_delta"] > 20
    assert simulated["simulated"]["score"] > simulated["baseline"]["score"]
    assert simulated["simulated"]["probability"] >= simulated["baseline"]["probability"]
    assert len(simulated["prescriptions"]) > 0
