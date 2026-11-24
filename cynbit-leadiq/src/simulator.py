"""
What-If Scenario Simulator and Lead Optimization Prescription Engine for Cynbit LeadIQ.
Enables sales and engineering teams to simulate requirement revisions and calculate score deltas.
"""

from typing import Dict, Any, List
from src.extraction import detect_service_categories
from src.scoring import compute_qualification_score
from src.prediction import ConversionPredictor


def simulate_lead_modifications(
    predictor: ConversionPredictor,
    base_lead: Dict[str, Any],
    new_inquiry_text: str,
    new_budget: str,
    new_timeline: str,
    new_company_size: str,
    new_role: str
) -> Dict[str, Any]:
    """
    Computes real-time comparative deltas between baseline lead parameters and a simulated scenario.
    Returns baseline metrics, simulated metrics, delta values, and prescriptive recommendations.
    """
    # 1. Evaluate baseline
    base_text = base_lead.get("inquiry_text", "")
    _, _, base_entities = detect_service_categories(base_text)
    base_score = compute_qualification_score(
        inquiry_text=base_text,
        budget_tier=base_lead.get("budget_tier", "Unspecified / Flexible"),
        timeline=base_lead.get("timeline", "Exploratory / No fixed date"),
        company_size=base_lead.get("company_size", "Not Specified"),
        email=base_lead.get("email", ""),
        role=base_lead.get("role", ""),
        source_channel=base_lead.get("source_channel", "Website Form"),
        detected_entities=base_entities
    )
    base_pred = predictor.predict_lead(
        inquiry_text=base_text,
        primary_category=base_lead.get("primary_category", "Other / Advisory"),
        budget_tier=base_lead.get("budget_tier", "Unspecified / Flexible"),
        timeline=base_lead.get("timeline", "Exploratory / No fixed date"),
        company_size=base_lead.get("company_size", "Not Specified"),
        source_channel=base_lead.get("source_channel", "Website Form"),
        qualification_score=base_score.total_score
    )

    # 2. Evaluate simulated scenario
    sim_cat, _, sim_entities = detect_service_categories(new_inquiry_text)
    sim_score = compute_qualification_score(
        inquiry_text=new_inquiry_text,
        budget_tier=new_budget,
        timeline=new_timeline,
        company_size=new_company_size,
        email=base_lead.get("email", ""),
        role=new_role,
        source_channel=base_lead.get("source_channel", "Website Form"),
        detected_entities=sim_entities
    )
    sim_pred = predictor.predict_lead(
        inquiry_text=new_inquiry_text,
        primary_category=sim_cat,
        budget_tier=new_budget,
        timeline=new_timeline,
        company_size=new_company_size,
        source_channel=base_lead.get("source_channel", "Website Form"),
        qualification_score=sim_score.total_score
    )

    score_delta = sim_score.total_score - base_score.total_score
    prob_delta = sim_pred.conversion_probability - base_pred.conversion_probability

    # Generate Optimization Prescriptions
    prescriptions: List[str] = []
    if "exploratory" in new_timeline.lower():
        prescriptions.append("Target a defined pilot sprint or proof-of-concept milestone within 30 days to raise urgency.")
    if "low" in new_budget.lower():
        prescriptions.append("Propose an initial phased MVP or discovery roadmap rather than an unviable full-scope custom build.")
    if len(sim_entities) < 2:
        prescriptions.append("Encourage the client to specify target cloud infrastructure (AWS/GCP) or existing codebase frameworks.")
    if sim_score.clarity_score < 15:
        prescriptions.append("Send a structured discovery scoping questionnaire to establish specific non-functional requirements.")

    return {
        "baseline": {
            "score": base_score.total_score,
            "tier": base_score.qualification_tier,
            "probability": base_pred.conversion_probability,
            "likelihood_label": base_pred.likelihood_label,
            "category": base_lead.get("primary_category")
        },
        "simulated": {
            "score": sim_score.total_score,
            "tier": sim_score.qualification_tier,
            "probability": sim_pred.conversion_probability,
            "likelihood_label": sim_pred.likelihood_label,
            "category": sim_cat,
            "detected_entities": sim_entities,
            "score_breakdown": sim_score
        },
        "deltas": {
            "score_delta": score_delta,
            "probability_delta": round(prob_delta, 4),
            "tier_changed": base_score.qualification_tier != sim_score.qualification_tier
        },
        "prescriptions": prescriptions if prescriptions else ["Lead exhibits strong optimization and requirements balance."]
    }
