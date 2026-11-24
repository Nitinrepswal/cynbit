"""
Preloaded synthetic demo leads for Cynbit LeadIQ.
Represents diverse service domains, urgency tiers, client maturity levels, and budget ranges.
Clearly labeled as synthetic benchmark records.
"""

from typing import List, Dict, Any
from src.extraction import detect_service_categories, extract_keywords_tf
from src.scoring import compute_qualification_score
from src.prediction import ConversionPredictor

DEMO_RAW_LEADS = [
    {
        "name": "Dr. Marcus Vance",
        "email": "m.vance@neurohealth-analytics.io",
        "company": "NeuroHealth Analytics Corp",
        "role": "VP of Data & AI",
        "inquiry_text": (
            "We are seeking an engineering partner to architect an enterprise predictive analytics and NLP pipeline "
            "for clinical documentation. Our internal team has structured tabular records and millions of clinical notes. "
            "We need custom PyTorch models with fine-tuned embeddings, containerized on AWS with HIPAA compliance. "
            "Timeline is immediate (< 2 weeks) and executive budget is secured."
        ),
        "budget_tier": "> $50,000 (Enterprise)",
        "timeline": "Immediate (< 2 weeks)",
        "company_size": "Enterprise (500+ employees)",
        "source_channel": "Referral / Partner",
        "lead_status": "Discovery Scheduled",
        "notes": "Met at AI Summit. High priority enterprise account."
    },
    {
        "name": "Elena Rostova",
        "email": "elena@finflow-systems.com",
        "company": "FinFlow Systems",
        "role": "Head of Product",
        "inquiry_text": (
            "Looking to build a multi-tenant B2B SaaS financial dashboard in Next.js and Python FastAPI. "
            "Requirements include Stripe billing integration, role-based access control, export to CSV/PDF, "
            "and real-time chart updates via WebSockets. We have wireframes ready in Figma and target kickoff within 4 weeks."
        ),
        "budget_tier": "$15,000 - $50,000 (High)",
        "timeline": "Short Term (1 - 2 months)",
        "company_size": "Mid-Market (51 - 500 employees)",
        "source_channel": "Website Form",
        "lead_status": "Qualified",
        "notes": "Shared Figma designs; technical fit verified."
    },
    {
        "name": "Jordan Bell",
        "email": "jordan.bell@apexlogistics.com",
        "company": "Apex Global Freight",
        "role": "Director of Logistics Technology",
        "inquiry_text": (
            "We currently process over 40,000 bills of lading per month manually. We urgently need an automated workflow "
            "pipeline combining OCR text extraction, automated validation against our ERP, and webhook sync to our NetSuite database. "
            "Goal is to reduce processing time by 75% before Q3."
        ),
        "budget_tier": "$15,000 - $50,000 (High)",
        "timeline": "Immediate (< 2 weeks)",
        "company_size": "Mid-Market (51 - 500 employees)",
        "source_channel": "Direct Inbound",
        "lead_status": "Proposal Sent",
        "notes": "Sent scoping proposal; awaiting procurement approval."
    },
    {
        "name": "Samantha Wu",
        "email": "swu@kineticsports.app",
        "company": "Kinetic Sports",
        "role": "Founder & CEO",
        "inquiry_text": (
            "Need a cross-platform mobile application in Flutter for iOS and Android. App connects to Bluetooth heart rate monitors, "
            "tracks workout metrics offline, and syncs when connected to cloud. Looking for experienced Flutter developers who can build "
            "and publish to App Store and Google Play."
        ),
        "budget_tier": "$15,000 - $50,000 (High)",
        "timeline": "Short Term (1 - 2 months)",
        "company_size": "Startup / Small Business (2 - 50 employees)",
        "source_channel": "Website Form",
        "lead_status": "Contacted",
        "notes": "Completed initial intro call; preparing technical scope."
    },
    {
        "name": "David Thorne",
        "email": "david.thorne@scalecloud.io",
        "company": "ScaleCloud Infrastructure",
        "role": "Principal DevOps Engineer",
        "inquiry_text": (
            "We require assistance migrating 12 microservices from dedicated virtual machines to managed Kubernetes (EKS) on AWS. "
            "Must establish automated CI/CD deployment pipelines using GitLab, Terraform provisioning, and Grafana monitoring."
        ),
        "budget_tier": "$15,000 - $50,000 (High)",
        "timeline": "Short Term (1 - 2 months)",
        "company_size": "Mid-Market (51 - 500 employees)",
        "source_channel": "Website Form",
        "lead_status": "New",
        "notes": ""
    },
    {
        "name": "Amina Kassam",
        "email": "amina@omnistore-retail.com",
        "company": "OmniStore Retail",
        "role": "E-Commerce Director",
        "inquiry_text": (
            "Exploring an AI chatbot for customer support queries on our Shopify store. We want to automate order status "
            "lookups and return requests. Would like an initial discussion regarding feasibility and approximate cost."
        ),
        "budget_tier": "$5,000 - $15,000 (Medium)",
        "timeline": "Short Term (1 - 2 months)",
        "company_size": "Startup / Small Business (2 - 50 employees)",
        "source_channel": "Website Form",
        "lead_status": "Contacted",
        "notes": "Requested Shopify API access to assess integration effort."
    },
    {
        "name": "Lucas Meyer",
        "email": "lucas@propdata-insights.de",
        "company": "PropData Insights",
        "role": "Lead Analyst",
        "inquiry_text": (
            "We need an automated web scraping system to monitor commercial real estate lease listings and store structured "
            "records in PostgreSQL. Would like scheduled weekly runs and error alerting."
        ),
        "budget_tier": "$5,000 - $15,000 (Medium)",
        "timeline": "Medium Term (3 - 6 months)",
        "company_size": "Startup / Small Business (2 - 50 employees)",
        "source_channel": "Website Form",
        "lead_status": "New",
        "notes": ""
    },
    {
        "name": "Tariq Mansoor",
        "email": "tariq.mansoor@gulfconsulting.ae",
        "company": "Gulf Advisory Partners",
        "role": "Managing Partner",
        "inquiry_text": (
            "Seeking technical advisory and architectural review for our client undergoing digital transformation. "
            "Need senior cloud and AI consulting support for 2-3 months."
        ),
        "budget_tier": "$15,000 - $50,000 (High)",
        "timeline": "Short Term (1 - 2 months)",
        "company_size": "Mid-Market (51 - 500 employees)",
        "source_channel": "Referral / Partner",
        "lead_status": "New",
        "notes": ""
    },
    {
        "name": "Kevin Miller",
        "email": "kevin_m_builder@gmail.com",
        "company": "Solo Project",
        "role": "Individual",
        "inquiry_text": (
            "I have an idea for an on-demand lawn mowing application like Uber. Looking for a developer who can build "
            "the mobile app and website for cheap or for equity. Let me know if interested."
        ),
        "budget_tier": "< $5,000 (Low)",
        "timeline": "Exploratory / No fixed date",
        "company_size": "Solo Founder / Individual",
        "source_channel": "Website Form",
        "lead_status": "Disqualified",
        "notes": "Unfunded individual idea; outside target customer profile."
    },
    {
        "name": "Chloe Bennett",
        "email": "chloe.b@lifestyle-blog.me",
        "company": "Personal Blog",
        "role": "Blogger",
        "inquiry_text": (
            "Hi, can someone tell me how much it costs to make a simple website? Just need a few pages."
        ),
        "budget_tier": "< $5,000 (Low)",
        "timeline": "Exploratory / No fixed date",
        "company_size": "Solo Founder / Individual",
        "source_channel": "Website Form",
        "lead_status": "Disqualified",
        "notes": "Referred to standard website builders."
    },
    {
        "name": "Anand Patel",
        "email": "anand.quant.student@yahoo.com",
        "company": "Self",
        "role": "Student",
        "inquiry_text": (
            "Can you help me train a deep learning LSTM model to predict crypto price movements? "
            "I have Python scripts but need someone to debug them for free or low cost."
        ),
        "budget_tier": "< $5,000 (Low)",
        "timeline": "Exploratory / No fixed date",
        "company_size": "Solo Founder / Individual",
        "source_channel": "Website Form",
        "lead_status": "Disqualified",
        "notes": "Academic homework/student inquiry."
    },
    {
        "name": "Rachel Adams",
        "email": "info@summitcorp-demo.com",
        "company": "Summit Corp",
        "role": "Unknown",
        "inquiry_text": (
            "Please send your general company brochure and service pricing overview."
        ),
        "budget_tier": "Unspecified / Flexible",
        "timeline": "Exploratory / No fixed date",
        "company_size": "Not Specified",
        "source_channel": "Website Form",
        "lead_status": "New",
        "notes": ""
    }
]


def generate_processed_demo_leads(predictor: ConversionPredictor) -> List[Dict[str, Any]]:
    """
    Processes the raw demo leads through NLP extraction, rule qualification,
    and ML prediction pipeline to create fully hydrated database records.
    """
    processed = []
    for item in DEMO_RAW_LEADS:
        primary_cat, cat_scores, detected_entities = detect_service_categories(item["inquiry_text"])
        kw_scores = extract_keywords_tf(item["inquiry_text"], top_n=6)
        extracted_kws = ", ".join([f"{w} ({s})" for w, s in kw_scores])

        score_res = compute_qualification_score(
            inquiry_text=item["inquiry_text"],
            budget_tier=item["budget_tier"],
            timeline=item["timeline"],
            company_size=item["company_size"],
            email=item["email"],
            role=item["role"],
            source_channel=item["source_channel"],
            detected_entities=detected_entities
        )

        pred_res = predictor.predict_lead(
            inquiry_text=item["inquiry_text"],
            primary_category=primary_cat,
            budget_tier=item["budget_tier"],
            timeline=item["timeline"],
            company_size=item["company_size"],
            source_channel=item["source_channel"],
            qualification_score=score_res.total_score
        )

        reasons_formatted = "\n".join(score_res.scoring_reasons)

        record = {
            "name": item["name"],
            "email": item["email"],
            "company": item["company"],
            "role": item["role"],
            "inquiry_text": item["inquiry_text"],
            "budget_tier": item["budget_tier"],
            "timeline": item["timeline"],
            "company_size": item["company_size"],
            "source_channel": item["source_channel"],
            "primary_category": primary_cat,
            "extracted_keywords": extracted_kws,
            "qualification_score": score_res.total_score,
            "qualification_tier": score_res.qualification_tier,
            "score_reasons": reasons_formatted,
            "conversion_probability": pred_res.conversion_probability,
            "ml_prediction_label": pred_res.likelihood_label,
            "lead_status": item["lead_status"],
            "notes": item["notes"]
        }
        processed.append(record)

    return processed
