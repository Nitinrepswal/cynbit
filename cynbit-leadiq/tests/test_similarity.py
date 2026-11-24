"""
Unit tests for semantic similarity engine.
"""

import pandas as pd
from src.similarity import find_similar_leads


def test_find_similar_leads_returns_relevant_matches():
    corpus = pd.DataFrame([
        {
            "id": 1,
            "name": "Alice",
            "company": "AI Health Corp",
            "inquiry_text": "We want to train PyTorch computer vision models on AWS Kubernetes for medical imaging.",
            "primary_category": "AI/ML Solutions",
            "qualification_tier": "Hot",
            "qualification_score": 90,
            "conversion_probability": 0.88,
            "lead_status": "Closed - Won"
        },
        {
            "id": 2,
            "name": "Bob",
            "company": "WebShop Co",
            "inquiry_text": "Need a simple WordPress blog and landing page for marketing.",
            "primary_category": "Web Development",
            "qualification_tier": "Cold",
            "qualification_score": 25,
            "conversion_probability": 0.15,
            "lead_status": "Closed - Lost"
        }
    ])

    query = "Looking for deep learning computer vision and PyTorch engineering on cloud AWS."
    matches = find_similar_leads(query, corpus, current_lead_id=None, top_k=2)

    assert len(matches) > 0
    # First match should be Alice due to high semantic term overlap
    assert matches[0]["id"] == 1
    assert matches[0]["similarity_score"] > matches[1]["similarity_score"]
    assert any("pytorch" in t or "vision" in t or "aws" in t for t in matches[0]["shared_keywords"])


def test_find_similar_leads_excludes_self():
    corpus = pd.DataFrame([
        {
            "id": 10,
            "name": "Lead 10",
            "company": "Company 10",
            "inquiry_text": "Flutter mobile app development for fitness tracker.",
            "primary_category": "Mobile Apps",
            "qualification_tier": "Warm",
            "qualification_score": 60,
            "conversion_probability": 0.55,
            "lead_status": "Qualified"
        }
    ])
    matches = find_similar_leads("Flutter mobile app", corpus, current_lead_id=10, top_k=2)
    assert len(matches) == 0
