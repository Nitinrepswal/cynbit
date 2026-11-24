"""
Synthetic dataset generator and ML conversion prediction pipeline trainer for Cynbit LeadIQ.
Uses Scikit-learn (TF-IDF + Categorical OneHot + Numeric Scaler -> Logistic Regression).
Strictly offline, reproducible with RANDOM_STATE=42, and prevents data leakage.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    METRICS_FILE_PATH,
    MODEL_FILE_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TRAINING_DATA_PATH,
)
from src.extraction import detect_service_categories
from src.preprocessing import clean_text
from src.scoring import compute_qualification_score

# Explicit synthetic text templates representing authentic client inquiries
SYNTHETIC_TEMPLATES = [
    # High Conversion / Enterprise / AI/ML
    (
        "We are looking to develop an enterprise-grade predictive analytics platform using PyTorch and FastAPI. "
        "Our data science team needs external expertise to build scalable inference pipelines on AWS Kubernetes. "
        "We have historical data ready and need an MVP delivered within 6 weeks. Budget is approved.",
        "AI/ML Solutions", "> $50,000 (Enterprise)", "Immediate (< 2 weeks)", "Enterprise (500+ employees)", "Referral / Partner", "VP of Engineering", "enterprise_vp@healthtech.com", 1
    ),
    (
        "Need an end-to-end customer churn forecasting engine integrated with our Snowflake warehouse and Salesforce CRM. "
        "The model must run daily batch predictions with feature attribution and an executive dashboard. "
        "Kickoff scheduled for next month with dedicated budget.",
        "AI/ML Solutions", "$15,000 - $50,000 (High)", "Short Term (1 - 2 months)", "Mid-Market (51 - 500 employees)", "Website Form", "Director of Analytics", "sarah@retailcorp.io", 1
    ),
    (
        "Looking for an AI partner to develop an internal document search and automated summarization tool for legal contracts. "
        "Must support PDF extraction, vector embeddings, fine-tuned RAG pipeline, and local LLM deployment with privacy compliance.",
        "AI/ML Solutions", "> $50,000 (Enterprise)", "Short Term (1 - 2 months)", "Enterprise (500+ employees)", "Direct Inbound", "Chief Information Officer", "legal_tech@firm.com", 1
    ),
    (
        "We want to build a computer vision defect detection system on an assembly line. "
        "Edge cameras will capture image frames and detect micro-scratches in real time using TensorRT. "
        "Need complete architecture design and edge deployment.",
        "AI/ML Solutions", "$15,000 - $50,000 (High)", "Immediate (< 2 weeks)", "Mid-Market (51 - 500 employees)", "Referral / Partner", "Head of Operations", "j.adams@mfg-solutions.com", 1
    ),

    # High Conversion / Web & SaaS
    (
        "Seeking an experienced engineering team to build a multi-tenant B2B SaaS web portal. "
        "Frontend in React/Next.js and backend in Python FastAPI with Stripe billing integration and role-based access. "
        "Wireframes and PRD are 90% complete.",
        "Web Development", "$15,000 - $50,000 (High)", "Short Term (1 - 2 months)", "Startup / Small Business (2 - 50 employees)", "Website Form", "CTO & Co-founder", "alex@fintechsaas.co", 1
    ),
    (
        "We are modernizing our legacy internal portal into a responsive web application. "
        "Requires microservices architecture, GraphQL API gateway, SSO with Okta, and real-time WebSocket notifications. "
        "Executive approval obtained.",
        "Web Development", "> $50,000 (Enterprise)", "Immediate (< 2 weeks)", "Enterprise (500+ employees)", "Direct Inbound", "VP of Digital", "mark.r@logistics-global.com", 1
    ),

    # High Conversion / Mobile Apps
    (
        "We need a cross-platform mobile application in Flutter for iOS and Android. "
        "App requires offline-first sync, Bluetooth IoT device pairing, payment gateway, and biometrics login. "
        "Designs in Figma are finalized.",
        "Mobile Apps", "$15,000 - $50,000 (High)", "Immediate (< 2 weeks)", "Mid-Market (51 - 500 employees)", "Website Form", "Product Manager", "linda@medtrack.org", 1
    ),

    # High Conversion / Automation & Cloud
    (
        "Looking to automate complex invoice processing workflows using OCR and custom RPA connectors to NetSuite. "
        "Currently handling 50,000 invoices manually per quarter. Seeking rapid implementation with high ROI.",
        "Automation & Workflows", "$15,000 - $50,000 (High)", "Immediate (< 2 weeks)", "Mid-Market (51 - 500 employees)", "Referral / Partner", "Finance Operations Lead", "kevin@globalfreight.com", 1
    ),
    (
        "Need a DevOps specialist to migrate our multi-tier applications from on-premises servers to AWS. "
        "Must implement Terraform infrastructure-as-code, Dockerized CI/CD pipelines in GitLab, and Prometheus/Grafana monitoring.",
        "Cloud & DevOps", "$15,000 - $50,000 (High)", "Short Term (1 - 2 months)", "Mid-Market (51 - 500 employees)", "Website Form", "Infrastructure Lead", "devops@cloudservices.net", 1
    ),

    # Moderate / Warm Leads (Mixed Conversion)
    (
        "Interested in discussing an AI chatbot for customer support queries on our e-commerce store. "
        "We have around 2,000 inquiries daily. Want to understand feasibility, implementation timeline, and cost estimates.",
        "AI/ML Solutions", "$5,000 - $15,000 (Medium)", "Short Term (1 - 2 months)", "Startup / Small Business (2 - 50 employees)", "Website Form", "Founder", "david@shoponline.io", 1
    ),
    (
        "We are planning a mobile application for local fitness trainers to schedule sessions and accept client bookings. "
        "Need basic MVP scope and ballpark quote.",
        "Mobile Apps", "$5,000 - $15,000 (Medium)", "Medium Term (3 - 6 months)", "Startup / Small Business (2 - 50 employees)", "Website Form", "Owner", "carlos@fitcoach.me", 0
    ),
    (
        "Exploring automated scraping and data collection from public real estate listings. "
        "Need weekly reports formatted as CSV or database sync. Still determining internal budget.",
        "Automation & Workflows", "$5,000 - $15,000 (Medium)", "Exploratory / No fixed date", "Startup / Small Business (2 - 50 employees)", "Organic Search", "Marketing Manager", "elena@realtydata.net", 0
    ),
    (
        "Looking for cloud optimization advice to lower our monthly Google Cloud Platform bills. "
        "Running several BigQuery queries and Kubernetes clusters. Need an initial architectural audit.",
        "Cloud & DevOps", "$5,000 - $15,000 (Medium)", "Short Term (1 - 2 months)", "Startup / Small Business (2 - 50 employees)", "Website Form", "Lead Developer", "tim@techstartup.co", 1
    ),
    (
        "We want to redesign our company website and migrate to WordPress or Next.js. "
        "Approximately 15 pages with a blog and career portal. Timeline is flexible over next quarter.",
        "Web Development", "$5,000 - $15,000 (Medium)", "Medium Term (3 - 6 months)", "Startup / Small Business (2 - 50 employees)", "Website Form", "Brand Manager", "rachel@designstudio.org", 0
    ),

    # Low Conversion / Cold Leads
    (
        "Hi, I have an idea for an app like Uber but for dog walking. "
        "I need someone to build the entire app and website. I am looking for a technical cofounder or cheap developer.",
        "Mobile Apps", "< $5,000 (Low)", "Exploratory / No fixed date", "Solo Founder / Individual", "Website Form", "Unknown", "john1989@gmail.com", 0
    ),
    (
        "Can you build me a website? How much will it cost?",
        "Web Development", "< $5,000 (Low)", "Exploratory / No fixed date", "Solo Founder / Individual", "Website Form", "Unknown", "student_user@yahoo.com", 0
    ),
    (
        "Looking for free consulting or guidance on how to train machine learning models for stock market prediction.",
        "AI/ML Solutions", "< $5,000 (Low)", "Exploratory / No fixed date", "Solo Founder / Individual", "Website Form", "Student", "trader99@hotmail.com", 0
    ),
    (
        "Need someone to fix a small script error on my server. Will pay $100.",
        "Cloud & DevOps", "< $5,000 (Low)", "Immediate (< 2 weeks)", "Solo Founder / Individual", "Website Form", "Unknown", "freelance_gig@gmail.com", 0
    ),
    (
        "Just browsing your services. Please send a brochure and standard price list for all services.",
        "Other / Advisory", "Unspecified / Flexible", "Exploratory / No fixed date", "Not Specified", "Website Form", "Unknown", "info@genericcorp.xyz", 0
    ),
    (
        "Interested in learning more about your AI offerings. No immediate project currently.",
        "AI/ML Solutions", "Unspecified / Flexible", "Exploratory / No fixed date", "Not Specified", "Marketing Campaign", "Unknown", "researcher@university.edu", 0
    ),
]


def generate_synthetic_dataset(num_records: int = 400, random_seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a reproducible, explicitly synthetic dataset for ML pipeline training and benchmarking.
    Includes realistic feature variation, deliberate noise, and ground-truth conversion outcomes.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    rows = []
    base_n = len(SYNTHETIC_TEMPLATES)

    for i in range(num_records):
        template = SYNTHETIC_TEMPLATES[i % base_n]
        text_base, category_base, budget_base, timeline_base, size_base, channel_base, role_base, email_base, label_base = template

        # Add natural linguistic variations and modifiers to avoid identical duplicates
        variations = [
            f"{text_base} (Ref #{1000 + i})",
            f"Regarding our Q{(i % 4) + 1} initiatives: {text_base}",
            f"Hello Cynbit Team, {text_base} Looking forward to discussing.",
            f"{text_base} Please provide relevant case studies.",
            f"{text_base}",
        ]
        inquiry_text = random.choice(variations)

        # Minor probabilistic variations in budget and timeline
        budget = budget_base
        timeline = timeline_base
        company_size = size_base
        channel = channel_base
        role = role_base
        email = f"lead_{i}_{email_base}"

        # Detect categories and compute qualification score
        primary_cat, scores, detected_entities = detect_service_categories(inquiry_text)
        score_res = compute_qualification_score(
            inquiry_text=inquiry_text,
            budget_tier=budget,
            timeline=timeline,
            company_size=company_size,
            email=email,
            role=role,
            source_channel=channel,
            detected_entities=detected_entities
        )

        # Probabilistic conversion label based on latent score + logistic curve + noise
        latent_z = (score_res.total_score - 52.0) / 14.0
        prob = 1.0 / (1.0 + np.exp(-latent_z))
        noise = np.random.normal(0, 0.18)
        noisy_prob = np.clip(prob + noise, 0.0, 1.0)
        converted = 1 if noisy_prob >= 0.50 else 0

        rows.append({
            "lead_id": f"SYN-{10000 + i}",
            "name": f"Lead Contact {i + 1}",
            "email": email,
            "company": f"Synthetic Org {i + 1}",
            "role": role,
            "inquiry_text": inquiry_text,
            "cleaned_text": clean_text(inquiry_text),
            "budget_tier": budget,
            "timeline": timeline,
            "company_size": company_size,
            "source_channel": channel,
            "primary_category": primary_cat,
            "word_count": len(inquiry_text.split()),
            "qualification_score": score_res.total_score,
            "qualification_tier": score_res.qualification_tier,
            "converted": converted
        })

    df = pd.DataFrame(rows)
    return df


def train_conversion_model(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_seed: int = RANDOM_STATE
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains a Scikit-Learn conversion prediction pipeline:
    - Text: TF-IDF Vectorizer
    - Categorical: One-Hot Encoder
    - Numerical: StandardScaler
    - Classifier: LogisticRegression with L2 regularization
    Strictly prevents data leakage by fitting transformers on train split only.
    """
    feature_cols = [
        "inquiry_text",
        "primary_category",
        "budget_tier",
        "timeline",
        "company_size",
        "source_channel",
        "qualification_score",
        "word_count"
    ]
    target_col = "converted"

    X = df[feature_cols]
    y = df[target_col].values

    # Stratified Train-Test Split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Column Transformers
    text_transformer = TfidfVectorizer(
        max_features=150,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2
    )

    categorical_features = [
        "primary_category",
        "budget_tier",
        "timeline",
        "company_size",
        "source_channel"
    ]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    numeric_features = ["qualification_score", "word_count"]
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "inquiry_text"),
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="drop"
    )

    # Classification Pipeline
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_seed
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Fit pipeline strictly on training data
    pipeline.fit(X_train, y_train)

    # Evaluate strictly on test split
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    # Extract Feature Importances (Logistic Regression coefficients)
    try:
        tfidf_names = list(pipeline.named_steps["preprocessor"].named_transformers_["text"].get_feature_names_out())
        cat_names = list(pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out())
        num_names = numeric_features
        all_feature_names = tfidf_names + cat_names + num_names
        coefficients = pipeline.named_steps["classifier"].coef_[0]

        feature_importance = [
            {"feature": name, "coefficient": round(float(coef), 4), "odds_ratio": round(float(np.exp(coef)), 4)}
            for name, coef in zip(all_feature_names, coefficients)
        ]
        # Sort top positive and top negative
        top_positive = sorted(feature_importance, key=lambda x: x["coefficient"], reverse=True)[:10]
        top_negative = sorted(feature_importance, key=lambda x: x["coefficient"])[:10]
    except Exception as e:
        top_positive = []
        top_negative = []

    metrics = {
        "model_type": "Logistic Regression with TF-IDF + Structured Features",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "random_seed": random_seed,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "positive_class_ratio_train": round(float(np.mean(y_train)), 3),
        "positive_class_ratio_test": round(float(np.mean(y_test)), 3),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negatives": cm[0][0],
            "false_positives": cm[0][1],
            "false_negatives": cm[1][0],
            "true_positives": cm[1][1]
        },
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "synthetic_data_disclaimer": (
            "NOTICE: This model was trained on reproducibly synthesized benchmark data. "
            "High test metrics reflect synthetic statistical fidelity and do not constitute "
            "evidence of real-world enterprise conversion accuracy."
        )
    }

    return pipeline, metrics


def build_and_save_artifacts() -> Dict[str, Any]:
    """Generates synthetic training dataset, trains pipeline, and persists artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Generate & save dataset
    df = generate_synthetic_dataset(num_records=450, random_seed=RANDOM_STATE)
    df.to_csv(TRAINING_DATA_PATH, index=False)

    # 2. Train model & compute metrics
    pipeline, metrics = train_conversion_model(df, test_size=0.20, random_seed=RANDOM_STATE)

    # 3. Persist pipeline & metadata
    joblib.dump(pipeline, MODEL_FILE_PATH)
    with open(METRICS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    print("Generating synthetic data and training Cynbit LeadIQ conversion model...")
    results = build_and_save_artifacts()
    print("Training Complete. Test Metrics:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
