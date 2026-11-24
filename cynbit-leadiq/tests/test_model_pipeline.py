"""
Integration tests for Cynbit LeadIQ ML conversion prediction pipeline.
"""

from src.prediction import ConversionPredictor, compare_qualification_and_ml


def test_conversion_predictor_inference():
    predictor = ConversionPredictor()
    assert predictor.is_ready()

    # Test High-Likelihood Profile
    res_high = predictor.predict_lead(
        inquiry_text=(
            "Looking to deploy enterprise PyTorch predictive models on AWS Kubernetes. "
            "Dedicated budget secured and timeline is immediate."
        ),
        primary_category="AI/ML Solutions",
        budget_tier="> $50,000 (Enterprise)",
        timeline="Immediate (< 2 weeks)",
        company_size="Enterprise (500+ employees)",
        source_channel="Referral / Partner",
        qualification_score=88
    )
    assert 0.0 <= res_high.conversion_probability <= 1.0
    assert res_high.predicted_class in (0, 1)
    assert len(res_high.top_positive_factors) > 0

    # Test Low-Likelihood Profile
    res_low = predictor.predict_lead(
        inquiry_text="Need cheap website or free advice.",
        primary_category="Web Development",
        budget_tier="< $5,000 (Low)",
        timeline="Exploratory / No fixed date",
        company_size="Solo Founder / Individual",
        source_channel="Website Form",
        qualification_score=22
    )
    assert 0.0 <= res_low.conversion_probability <= 1.0
    assert res_low.conversion_probability < res_high.conversion_probability


def test_compare_qualification_and_ml():
    comparison = compare_qualification_and_ml(qualification_score=85, conversion_prob=0.88)
    assert comparison["rule_tier"] == "High"
    assert comparison["ml_tier"] == "High"
    assert "Consensus" in comparison["analysis"]

    divergence = compare_qualification_and_ml(qualification_score=85, conversion_prob=0.20)
    assert divergence["rule_tier"] == "High"
    assert divergence["ml_tier"] == "Low"
    assert "Divergence" in divergence["analysis"]
