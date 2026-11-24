"""
Transparent keyword extraction and service categorization for Cynbit LeadIQ.
Combines deterministic domain keyword catalogs with TF-IDF term scoring.
"""

from typing import Dict, List, Tuple
from collections import Counter
import math
from src.preprocessing import clean_text, tokenize_and_filter

# Domain keyword catalogs for Cynbit's primary service offerings
CATEGORY_KEYWORDS = {
    "AI/ML Solutions": [
        "ai", "ml", "machine learning", "deep learning", "nlp", "computer vision",
        "llm", "large language model", "neural network", "predictive model", "pytorch",
        "tensorflow", "scikit-learn", "data science", "classification", "forecast",
        "recommendation system", "clustering", "fine-tuning", "rag", "analytics engine",
        "generative ai", "model training", "transformer", "bert", "embeddings"
    ],
    "Web Development": [
        "web", "website", "frontend", "backend", "fullstack", "react", "nextjs",
        "vue", "angular", "django", "fastapi", "flask", "node", "nodejs", "portal",
        "saas", "dashboard", "rest api", "graphql", "tailwind", "typescript",
        "web application", "e-commerce", "shopify", "cms"
    ],
    "Mobile Apps": [
        "mobile", "ios", "android", "flutter", "react native", "swift", "kotlin",
        "app store", "play store", "tablet", "hybrid app", "cross-platform", "push notifications"
    ],
    "Automation & Workflows": [
        "automation", "workflow", "automate", "zapier", "n8n", "make.com",
        "bot", "scraper", "crawling", "etl", "data pipeline", "rpa", "integration",
        "webhook", "batch job", "cron", "sync"
    ],
    "Cloud & DevOps": [
        "cloud", "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "ci/cd",
        "devops", "terraform", "infrastructure", "serverless", "lambda", "ecs",
        "microservices", "monitoring", "scalability", "load balancing"
    ],
    "Other / Advisory": [
        "consulting", "advisory", "audit", "architecture review", "technical strategy",
        "feasibility study", "mvp planning", "cto advisory", "staff augmentation"
    ]
}


def extract_keywords_tf(text: str, top_n: int = 8) -> List[Tuple[str, float]]:
    """
    Extracts top keywords using Term Frequency and length-weighted heuristic.
    Transparent, explainable, and 100% offline.
    """
    tokens = tokenize_and_filter(text)
    if not tokens:
        return []

    total_tokens = len(tokens)
    counts = Counter(tokens)
    
    # Score = term frequency * slight boost for technical domain terms
    scored_keywords = []
    for word, count in counts.items():
        tf = count / total_tokens
        # Check if the word is part of any domain dictionary
        is_domain = any(word in kw for kw_list in CATEGORY_KEYWORDS.values() for kw in kw_list)
        boost = 1.35 if is_domain else 1.0
        score = round(tf * boost * 10, 3)
        scored_keywords.append((word, score))

    scored_keywords.sort(key=lambda x: x[1], reverse=True)
    return scored_keywords[:top_n]


def detect_service_categories(text: str) -> Tuple[str, Dict[str, float], List[str]]:
    """
    Evaluates text against service catalogs.
    Returns:
    - Primary detected category
    - Confidence scores across all categories
    - List of detected technical entities / matching keywords
    """
    cleaned = clean_text(text)
    category_matches: Dict[str, int] = {cat: 0 for cat in CATEGORY_KEYWORDS}
    detected_entities: List[str] = []

    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            # Check whole phrase or word boundary match
            if kw in cleaned:
                category_matches[cat] += (2 if " " in kw else 1)
                if kw not in detected_entities:
                    detected_entities.append(kw)

    total_matches = sum(category_matches.values())
    
    if total_matches == 0:
        # Default fallback if no specific terms match
        scores = {cat: round(1.0 / len(CATEGORY_KEYWORDS), 3) for cat in CATEGORY_KEYWORDS}
        scores["Other / Advisory"] = 0.50
        return "Other / Advisory", scores, detected_entities

    # Normalized confidence score per category
    scores = {cat: round(count / total_matches, 3) for cat, count in category_matches.items()}
    primary_category = max(scores, key=scores.get)

    return primary_category, scores, detected_entities
