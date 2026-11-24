"""
Semantic similarity and nearest neighbor search module for Cynbit LeadIQ.
Uses TF-IDF embeddings and cosine similarity to discover analogous historical leads.
Runs 100% offline and provides concrete empirical reference cases.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import clean_text, tokenize_and_filter


def find_similar_leads(
    query_text: str,
    leads_df: pd.DataFrame,
    current_lead_id: Optional[int] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Finds the top K most semantically similar historical leads to the given inquiry text.
    Returns similarity percentages, shared keywords, and historical outcomes.
    """
    if leads_df.empty or not query_text.strip():
        return []

    # Exclude current lead from candidate pool if ID provided
    candidates = leads_df.copy()
    if current_lead_id is not None:
        candidates = candidates[candidates["id"] != current_lead_id]

    if candidates.empty:
        return []

    query_cleaned = clean_text(query_text)
    candidate_texts = [clean_text(t) for t in candidates["inquiry_text"].tolist()]
    all_texts = [query_cleaned] + candidate_texts

    # Fit TF-IDF on corpus
    vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    query_vec = tfidf_matrix[0:1]
    candidate_matrix = tfidf_matrix[1:]

    sim_scores = cosine_similarity(query_vec, candidate_matrix)[0]

    # Query tokens for term intersection
    query_tokens = set(tokenize_and_filter(query_cleaned))

    results = []
    candidates_list = candidates.to_dict(orient="records")

    # Pair candidates with scores
    scored = list(zip(candidates_list, sim_scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    for cand, score in scored[:top_k]:
        cand_tokens = set(tokenize_and_filter(clean_text(cand.get("inquiry_text", ""))))
        overlap = list(query_tokens.intersection(cand_tokens))[:4]

        results.append({
            "id": cand.get("id"),
            "name": cand.get("name"),
            "company": cand.get("company"),
            "primary_category": cand.get("primary_category"),
            "qualification_tier": cand.get("qualification_tier"),
            "qualification_score": cand.get("qualification_score"),
            "conversion_probability": cand.get("conversion_probability"),
            "lead_status": cand.get("lead_status"),
            "similarity_score": round(float(score) * 100, 1),
            "shared_keywords": overlap
        })

    return results
