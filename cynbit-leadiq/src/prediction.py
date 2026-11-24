"""
Inference engine and explainability interface for Cynbit LeadIQ ML conversion model.
Distinguishes empirical ML conversion probability from deterministic rule scoring.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from src.config import MODEL_FILE_PATH, METRICS_FILE_PATH
from src.schemas import PredictionResult


class ConversionPredictor:
    """Manages model loading, batch/single inference, and explainability."""

    def __init__(self, model_path: Path = MODEL_FILE_PATH):
        self.model_path = model_path
        self.pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads fitted pipeline from disk if available."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception as e:
                self.pipeline = None
        else:
            self.pipeline = None

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def predict_lead(
        self,
        inquiry_text: str,
        primary_category: str,
        budget_tier: str,
        timeline: str,
        company_size: str,
        source_channel: str,
        qualification_score: int
    ) -> PredictionResult:
        """
        Runs ML inference for a single lead.
        Returns conversion probability, predicted class, and factor explanation.
        """
        if not self.is_ready():
            # Graceful fallback baseline if model has not been trained yet
            return PredictionResult(
                conversion_probability=0.50,
                predicted_class=0,
                likelihood_label="Uncalibrated (Model Not Trained)",
                top_positive_factors=["Model artifact pending training"],
                top_negative_factors=[],
                model_version="Fallback Baseline v0"
            )

        # Prepare single-row DataFrame matching training schema
        word_count = len(inquiry_text.split())
        input_df = pd.DataFrame([{
            "inquiry_text": inquiry_text,
            "primary_category": primary_category,
            "budget_tier": budget_tier,
            "timeline": timeline,
            "company_size": company_size,
            "source_channel": source_channel,
            "qualification_score": qualification_score,
            "word_count": word_count
        }])

        prob = float(self.pipeline.predict_proba(input_df)[0][1])
        pred_class = int(prob >= 0.50)

        if prob >= 0.65:
            likelihood_label = "High Conversion Likelihood"
        elif prob >= 0.35:
            likelihood_label = "Moderate Conversion Likelihood"
        else:
            likelihood_label = "Low Conversion Likelihood"

        # Explainability: Extract key active features
        positive_factors = []
        negative_factors = []

        if qualification_score >= 70:
            positive_factors.append(f"Strong qualification score ({qualification_score}/100)")
        elif qualification_score < 40:
            negative_factors.append(f"Low qualification score ({qualification_score}/100)")

        if "enterprise" in budget_tier.lower() or "high" in budget_tier.lower():
            positive_factors.append(f"High-tier budget commitment ({budget_tier})")
        elif "low" in budget_tier.lower():
            negative_factors.append(f"Sub-optimal budget tier ({budget_tier})")

        if "immediate" in timeline.lower() or "short" in timeline.lower():
            positive_factors.append(f"Near-term delivery window ({timeline})")
        elif "exploratory" in timeline.lower():
            negative_factors.append("Exploratory purchase stage without firm launch deadline")

        if "enterprise" in company_size.lower() or "mid-market" in company_size.lower():
            positive_factors.append(f"Verified organization scale ({company_size})")

        if "referral" in source_channel.lower() or "inbound" in source_channel.lower():
            positive_factors.append(f"High-intent acquisition channel ({source_channel})")

        return PredictionResult(
            conversion_probability=round(prob, 4),
            predicted_class=pred_class,
            likelihood_label=likelihood_label,
            top_positive_factors=positive_factors if positive_factors else ["Baseline inquiry profile"],
            top_negative_factors=negative_factors if negative_factors else ["None identified"],
            model_version="LogisticRegression-TFIDF-v1"
        )


def compare_qualification_and_ml(qualification_score: int, conversion_prob: float) -> Dict[str, str]:
    """
    Explains the strategic divergence or agreement between deterministic rule score
    and empirical ML conversion probability.
    """
    qual_tier = "High" if qualification_score >= 70 else ("Medium" if qualification_score >= 40 else "Low")
    ml_tier = "High" if conversion_prob >= 0.65 else ("Medium" if conversion_prob >= 0.35 else "Low")

    if qual_tier == "High" and ml_tier == "High":
        insight = (
            "Consensus: Strong Rule Match & High ML Probability. "
            "Both strategic criteria (budget, organization size) and NLP text patterns indicate an ideal conversion candidate."
        )
    elif qual_tier == "High" and ml_tier in ("Medium", "Low"):
        insight = (
            "Divergence: High Strategic Fit, but Lower ML Conversion Likelihood. "
            "While corporate size and stated budget are favorable, inquiry semantics suggest tentative or early exploratory phrasing."
        )
    elif qual_tier in ("Medium", "Low") and ml_tier == "High":
        insight = (
            "Divergence: Moderate Strategic Fit, but Strong Conversion Probability. "
            "Even with modest budget or company size, inquiry text exhibits high intent signals, urgency, and technical clarity."
        )
    else:
        insight = (
            "Consensus: Low Strategic Fit & Low ML Probability. "
            "Inquiry lacks clear budget commitment, urgent timeline, and specific scope alignment."
        )

    return {
        "rule_tier": qual_tier,
        "ml_tier": ml_tier,
        "analysis": insight
    }
