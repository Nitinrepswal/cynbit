"""
Unit tests for Cynbit LeadIQ rule-based qualification scoring.
Verifies score bounds, dimensions, classification thresholds, and reasons.
"""

from src.scoring import compute_qualification_score


def test_hot_lead_scoring():
    score_res = compute_qualification_score(
        inquiry_text=(
            "We have secured budget to build an enterprise predictive intelligence platform using PyTorch on AWS. "
            "Our data team has 5M records and we require an end-to-end production pipeline delivered within 6 weeks. "
            "Need immediate technical discovery."
        ),
        budget_tier="> $50,000 (Enterprise)",
        timeline="Immediate (< 2 weeks)",
        company_size="Enterprise (500+ employees)",
        email="cto@globalhealth.org",
        role="Chief Technology Officer",
        source_channel="Referral / Partner",
        detected_entities=["pytorch", "aws", "predictive model"]
    )
    assert score_res.total_score >= 70
    assert score_res.qualification_tier == "Hot"
    assert score_res.clarity_score > 0
    assert score_res.urgency_score == 20
    assert score_res.budget_score == 20
    assert len(score_res.scoring_reasons) > 0


def test_cold_lead_scoring():
    score_res = compute_qualification_score(
        inquiry_text="Need app",
        budget_tier="< $5,000 (Low)",
        timeline="Exploratory / No fixed date",
        company_size="Solo Founder / Individual",
        email="user123@gmail.com",
        role="Unknown",
        source_channel="Website Form",
        detected_entities=[]
    )
    assert score_res.total_score < 40
    assert score_res.qualification_tier == "Cold"
    assert score_res.clarity_score <= 4
    assert score_res.urgency_score <= 5
    assert score_res.budget_score <= 5


def test_warm_lead_scoring_boundary():
    score_res = compute_qualification_score(
        inquiry_text="We want to build a customer portal for our retail franchise with Next.js and Postgres.",
        budget_tier="$5,000 - $15,000 (Medium)",
        timeline="Short Term (1 - 2 months)",
        company_size="Startup / Small Business (2 - 50 employees)",
        email="contact@retailstore.com",
        role="Product Manager",
        source_channel="Website Form",
        detected_entities=["nextjs", "portal"]
    )
    assert 40 <= score_res.total_score < 70
    assert score_res.qualification_tier == "Warm"


def test_score_never_exceeds_100_or_subzero():
    # Extreme upper bound test
    score_res_upper = compute_qualification_score(
        inquiry_text="Detailed inquiry " * 50,
        budget_tier="> $50,000 (Enterprise)",
        timeline="Immediate (< 2 weeks)",
        company_size="Enterprise (500+ employees)",
        email="vp@enterprise.com",
        role="VP of Engineering",
        source_channel="Referral / Partner",
        detected_entities=["ai", "cloud", "docker", "kubernetes", "pytorch"]
    )
    assert 0 <= score_res_upper.total_score <= 100

    # Extreme lower bound test
    score_res_lower = compute_qualification_score(
        inquiry_text="",
        budget_tier="< $5,000 (Low)",
        timeline="Exploratory / No fixed date",
        company_size="Solo Founder / Individual",
        email="x@gmail.com",
        role="",
        source_channel="",
        detected_entities=[]
    )
    assert 0 <= score_res_lower.total_score <= 100
