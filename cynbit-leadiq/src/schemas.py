"""
Data schemas and validation models for Cynbit LeadIQ.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, EmailStr, field_validator


class LeadInput(BaseModel):
    """Raw incoming lead input validation model."""
    name: str = Field(..., min_length=2, max_length=100, description="Full name or primary contact name")
    email: str = Field(..., description="Business or personal email address")
    company: Optional[str] = Field(default="Unspecified", max_length=120, description="Organization name")
    role: Optional[str] = Field(default="Unknown", max_length=80, description="Contact person role or title")
    inquiry_text: str = Field(..., min_length=10, max_length=5000, description="Details of the project inquiry")
    budget_tier: str = Field(default="Unspecified / Flexible", description="Stated or estimated budget level")
    timeline: str = Field(default="Exploratory / No fixed date", description="Stated delivery or start urgency")
    company_size: str = Field(default="Not Specified", description="Estimated company size category")
    source_channel: str = Field(default="Website Form", description="Origin of lead, e.g. Website, Referral, Inbound")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        v = v.strip()
        if "@" not in v or "." not in v.split("@")[-1] or len(v) < 5:
            raise ValueError("Invalid email format. Must contain an '@' and a domain suffix.")
        return v

    @field_validator("inquiry_text")
    @classmethod
    def sanitize_inquiry(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 10:
            raise ValueError("Inquiry text must be at least 10 characters long to provide sufficient context.")
        return cleaned


class NLPAnalysisResult(BaseModel):
    """NLP analysis output for a lead."""
    cleaned_text: str
    extracted_keywords: List[str]
    primary_category: str
    category_scores: Dict[str, float]
    detected_tech_entities: List[str]
    word_count: int


class ScoreBreakdown(BaseModel):
    """Transparent rule-based qualification score components."""
    clarity_score: int
    urgency_score: int
    budget_score: int
    company_size_score: int
    scope_depth_score: int
    engagement_score: int
    total_score: int
    qualification_tier: str  # 'Hot', 'Warm', 'Cold'
    scoring_reasons: List[str]


class PredictionResult(BaseModel):
    """Machine learning conversion prediction output."""
    conversion_probability: float
    predicted_class: int  # 1 (Convert) or 0 (No convert)
    likelihood_label: str  # 'High', 'Moderate', 'Low'
    top_positive_factors: List[str]
    top_negative_factors: List[str]
    model_version: str


class LeadRecord(BaseModel):
    """Full database representation of an evaluated lead."""
    id: Optional[int] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    name: str
    email: str
    company: str
    role: str
    inquiry_text: str
    budget_tier: str
    timeline: str
    company_size: str
    source_channel: str
    primary_category: str
    extracted_keywords: str  # Comma-separated
    qualification_score: int
    qualification_tier: str
    score_reasons: str  # JSON or newline-separated
    conversion_probability: float
    ml_prediction_label: str
    lead_status: str = "New"
    notes: Optional[str] = ""
