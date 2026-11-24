"""
Cynbit LeadIQ - Configuration & System Constants
Defines thresholds, service categories, weights, and relative paths.
"""

from pathlib import Path
from typing import Dict, List

# Base Paths (Relative to repository root)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_PATH = DATA_DIR / "cynbit_leads.db"
MODEL_FILE_PATH = MODELS_DIR / "conversion_pipeline.joblib"
METRICS_FILE_PATH = MODELS_DIR / "model_metadata.json"
TRAINING_DATA_PATH = DATA_DIR / "training_dataset.csv"
SAMPLE_LEADS_PATH = DATA_DIR / "synthetic_leads.csv"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Qualification Score Thresholds (Rule-Based 0-100)
TIER_HOT_THRESHOLD = 70
TIER_WARM_THRESHOLD = 40

# Service Categories
SERVICE_CATEGORIES: List[str] = [
    "AI/ML Solutions",
    "Web Development",
    "Mobile Apps",
    "Automation & Workflows",
    "Cloud & DevOps",
    "Other / Advisory",
]

# Budget Levels
BUDGET_TIERS: List[str] = [
    "< $5,000 (Low)",
    "$5,000 - $15,000 (Medium)",
    "$15,000 - $50,000 (High)",
    "> $50,000 (Enterprise)",
    "Unspecified / Flexible",
]

# Urgency / Timeline Options
TIMELINE_OPTIONS: List[str] = [
    "Immediate (< 2 weeks)",
    "Short Term (1 - 2 months)",
    "Medium Term (3 - 6 months)",
    "Exploratory / No fixed date",
]

# Company Size Options
COMPANY_SIZES: List[str] = [
    "Enterprise (500+ employees)",
    "Mid-Market (51 - 500 employees)",
    "Startup / Small Business (2 - 50 employees)",
    "Solo Founder / Individual",
    "Not Specified",
]

# Lead Status Options for CRM Workflow
LEAD_STATUS_OPTIONS: List[str] = [
    "New",
    "Contacted",
    "Qualified",
    "Discovery Scheduled",
    "Proposal Sent",
    "Closed - Won",
    "Closed - Lost",
    "Disqualified",
]

# Rule-Based Qualification Scoring Weight Limits
SCORE_MAX_COMPONENTS: Dict[str, int] = {
    "clarity": 20,
    "urgency": 20,
    "budget": 20,
    "company_size": 15,
    "scope_depth": 15,
    "engagement_channel": 10,
}
