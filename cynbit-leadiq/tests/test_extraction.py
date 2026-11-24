"""
Unit tests for Cynbit LeadIQ keyword extraction and categorization.
"""

from src.extraction import detect_service_categories, extract_keywords_tf


def test_detect_ai_ml_category():
    text = "We want to train a deep learning PyTorch model for computer vision anomaly detection and NLP embeddings."
    category, scores, entities = detect_service_categories(text)
    assert category == "AI/ML Solutions"
    assert "pytorch" in entities or "computer vision" in entities or "deep learning" in entities
    assert scores["AI/ML Solutions"] > scores["Web Development"]


def test_detect_web_development_category():
    text = "Looking for frontend engineering in React and Next.js with a Python FastAPI backend and Tailwind styling."
    category, scores, entities = detect_service_categories(text)
    assert category == "Web Development"
    assert "react" in entities or "nextjs" in entities


def test_detect_mobile_apps_category():
    text = "Need a cross-platform mobile application in Flutter for iOS and Android with offline sync and push notifications."
    category, scores, entities = detect_service_categories(text)
    assert category == "Mobile Apps"
    assert "flutter" in entities or "mobile" in entities or "ios" in entities


def test_detect_automation_category():
    text = "Automate our invoice approval workflow using custom RPA scripts, webhooks, and ETL pipeline."
    category, scores, entities = detect_service_categories(text)
    assert category == "Automation & Workflows"
    assert "automation" in entities or "workflow" in entities or "etl" in entities


def test_detect_cloud_devops_category():
    text = "Migrate 10 microservices from on-prem to AWS Kubernetes using Docker, Terraform, and CI/CD pipelines."
    category, scores, entities = detect_service_categories(text)
    assert category == "Cloud & DevOps"
    assert "kubernetes" in entities or "aws" in entities or "docker" in entities


def test_detect_other_fallback():
    text = "General inquiry regarding consulting rates and corporate strategy."
    category, scores, entities = detect_service_categories(text)
    assert category == "Other / Advisory"


def test_extract_keywords_tf():
    text = "Kubernetes Docker AWS cloud deployment infrastructure Terraform"
    kws = extract_keywords_tf(text, top_n=5)
    extracted_words = [w for w, _ in kws]
    assert len(kws) <= 5
    assert any(term in extracted_words for term in ["kubernetes", "docker", "aws", "cloud", "terraform"])
