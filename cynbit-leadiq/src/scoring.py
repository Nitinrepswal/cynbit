"""
Explainable, rule-based lead qualification scoring engine for Cynbit LeadIQ.
Assigns a deterministic score from 0 to 100 based on explicit business rules.
DO NOT confuse with ML conversion prediction (which estimates empirical conversion probability).
"""

from typing import Dict, Any, List
from src.config import TIER_HOT_THRESHOLD, TIER_WARM_THRESHOLD
from src.schemas import ScoreBreakdown
from src.preprocessing import clean_text

GENERIC_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
    "mail.com", "aol.com", "proton.me", "protonmail.com"
}

DECISION_MAKER_TITLES = {
    "cto", "cio", "ceo", "cpo", "founder", "co-founder", "vp", "vice president",
    "director", "head of", "lead", "architect", "engineering manager", "partner"
}


def compute_qualification_score(
    inquiry_text: str,
    budget_tier: str,
    timeline: str,
    company_size: str,
    email: str,
    role: str,
    source_channel: str,
    detected_entities: List[str]
) -> ScoreBreakdown:
    """
    Computes an explainable 0-100 qualification score across 6 transparent dimensions:
    1. Requirement Clarity (20 pts)
    2. Urgency & Timeline (20 pts)
    3. Budget Availability (20 pts)
    4. Business Size & Maturity (15 pts)
    5. Project Scope & Technical Depth (15 pts)
    6. Engagement & Stakeholder Quality (10 pts)
    """
    reasons: List[str] = []
    
    # --- 1. Requirement Clarity (Max 20) ---
    cleaned = clean_text(inquiry_text)
    word_count = len(cleaned.split())
    clarity_score = 0
    
    if word_count >= 70:
        clarity_score = 20
        reasons.append("+20 pts [Clarity]: Highly detailed project description with comprehensive specifications.")
    elif word_count >= 35:
        clarity_score = 15
        reasons.append("+15 pts [Clarity]: Clear inquiry with actionable business requirements.")
    elif word_count >= 15:
        clarity_score = 9
        reasons.append("+9 pts [Clarity]: Moderate detail; baseline requirements outlined.")
    else:
        clarity_score = 4
        reasons.append("+4 pts [Clarity]: Very brief or vague inquiry; requires substantial discovery clarification.")

    # --- 2. Urgency & Timeline (Max 20) ---
    urgency_score = 0
    t_lower = timeline.lower()
    if "immediate" in t_lower or "< 2 weeks" in t_lower:
        urgency_score = 20
        reasons.append("+20 pts [Urgency]: High immediate timeline (< 2 weeks) indicates urgent purchasing intent.")
    elif "short term" in t_lower or "1 - 2 months" in t_lower:
        urgency_score = 15
        reasons.append("+15 pts [Urgency]: Short-term kickoff target (1-2 months); active evaluation underway.")
    elif "medium term" in t_lower or "3 - 6 months" in t_lower:
        urgency_score = 8
        reasons.append("+8 pts [Urgency]: Medium-term milestone (3-6 months); planned roadmap.")
    else:
        urgency_score = 3
        reasons.append("+3 pts [Urgency]: Exploratory or uncommitted timeline; lower immediate priority.")

    # --- 3. Budget Availability (Max 20) ---
    budget_score = 0
    b_lower = budget_tier.lower()
    if "> $50,000" in b_lower or "enterprise" in b_lower:
        budget_score = 20
        reasons.append("+20 pts [Budget]: Enterprise-tier budget (> $50k) matches high-value engagement criteria.")
    elif "$15,000 - $50,000" in b_lower or "high" in b_lower:
        budget_score = 16
        reasons.append("+16 pts [Budget]: Substantial project budget ($15k-$50k) well-suited for custom delivery.")
    elif "$5,000 - $15,000" in b_lower or "medium" in b_lower:
        budget_score = 10
        reasons.append("+10 pts [Budget]: Moderate budget ($5k-$15k) feasible for an initial MVP or pilot build.")
    elif "< $5,000" in b_lower or "low" in b_lower:
        budget_score = 4
        reasons.append("+4 pts [Budget]: Budget under $5k is below Cynbit's typical custom project threshold.")
    else:
        budget_score = 7
        reasons.append("+7 pts [Budget]: Flexible/Unspecified budget; financial alignment needs discovery validation.")

    # --- 4. Business Size & Maturity (Max 15) ---
    size_score = 0
    s_lower = company_size.lower()
    if "enterprise" in s_lower or "500+" in s_lower:
        size_score = 15
        reasons.append("+15 pts [Company Size]: Large enterprise organization with verified procurement capacity.")
    elif "mid-market" in s_lower or "51 - 500" in s_lower:
        size_score = 12
        reasons.append("+12 pts [Company Size]: Mid-market scale provides strong budget with faster decision cycles.")
    elif "startup" in s_lower or "small business" in s_lower:
        size_score = 8
        reasons.append("+8 pts [Company Size]: Startup / SMB with agile decision-making process.")
    elif "solo" in s_lower or "individual" in s_lower:
        size_score = 4
        reasons.append("+4 pts [Company Size]: Solo founder or individual client with single-point sponsor risk.")
    else:
        size_score = 5
        reasons.append("+5 pts [Company Size]: Company size not specified; neutral baseline assigned.")

    # --- 5. Project Scope & Technical Depth (Max 15) ---
    scope_score = 0
    entity_count = len(detected_entities)
    if entity_count >= 4:
        scope_score = 15
        reasons.append(f"+15 pts [Technical Depth]: Explicit technical entities recognized ({', '.join(detected_entities[:4])}).")
    elif entity_count >= 2:
        scope_score = 11
        reasons.append(f"+11 pts [Technical Depth]: Good domain alignment ({', '.join(detected_entities)}).")
    elif entity_count == 1:
        scope_score = 7
        reasons.append(f"+7 pts [Technical Depth]: Single domain focus ({detected_entities[0]}).")
    else:
        scope_score = 4
        reasons.append("+4 pts [Technical Depth]: Broad or general inquiry without explicit technology keywords.")

    # --- 6. Engagement & Stakeholder Quality (Max 10) ---
    engagement_score = 0
    
    # Email domain check
    email_domain = email.split("@")[-1].lower() if "@" in email else ""
    if email_domain and email_domain not in GENERIC_EMAIL_DOMAINS:
        engagement_score += 4
        reasons.append("+4 pts [Stakeholder]: Corporate email domain verified.")
    else:
        reasons.append("+0 pts [Stakeholder]: Generic or personal email domain used.")

    # Role seniority
    r_lower = role.lower()
    if any(dm in r_lower for dm in DECISION_MAKER_TITLES):
        engagement_score += 3
        reasons.append(f"+3 pts [Stakeholder]: Decision-maker seniority confirmed ({role}).")
    elif role and role.lower() != "unknown":
        engagement_score += 2
        reasons.append(f"+2 pts [Stakeholder]: Specified role identified ({role}).")
    else:
        reasons.append("+0 pts [Stakeholder]: Role unspecified.")

    # Channel intent
    sc_lower = source_channel.lower()
    if "referral" in sc_lower or "partner" in sc_lower:
        engagement_score += 3
        reasons.append("+3 pts [Channel]: High-trust referral or partner channel.")
    elif "inbound" in sc_lower or "website" in sc_lower:
        engagement_score += 2
        reasons.append("+2 pts [Channel]: Direct website inbound inquiry.")
    else:
        engagement_score += 1
        reasons.append("+1 pt [Channel]: Standard outbound or marketing campaign channel.")

    total_score = clarity_score + urgency_score + budget_score + size_score + scope_score + engagement_score
    total_score = max(0, min(100, total_score))

    # Determine Tier
    if total_score >= TIER_HOT_THRESHOLD:
        tier = "Hot"
    elif total_score >= TIER_WARM_THRESHOLD:
        tier = "Warm"
    else:
        tier = "Cold"

    return ScoreBreakdown(
        clarity_score=clarity_score,
        urgency_score=urgency_score,
        budget_score=budget_score,
        company_size_score=size_score,
        scope_depth_score=scope_score,
        engagement_score=engagement_score,
        total_score=total_score,
        qualification_tier=tier,
        scoring_reasons=reasons
    )
