"""
Preprocessing and data sanitization routines for Cynbit LeadIQ.
Runs completely offline without external network or language model downloads.
"""

import re
import html
from typing import Dict, Any, Tuple, Optional

# Standard offline English stopwords list
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "please", "thanks", "hello", "hi", "regards", "sincerely", "dear"
}


def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw inquiry text:
    - Decodes HTML entities
    - Removes URLs and excessive whitespace
    - Strips non-alphanumeric punctuation while preserving spaces
    - Converts to lowercase
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unescape HTML entities (e.g. &amp;, &lt;)
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove email addresses from inquiry body to prevent leakage
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    
    # Replace punctuation and special characters with spaces
    text = re.sub(r'[^a-zA-Z0-9\s/+#.-]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()


def tokenize_and_filter(text: str) -> list[str]:
    """Tokenize cleaned text and filter out English stopwords and short tokens."""
    tokens = re.findall(r'\b[a-z0-9+#.-]{2,}\b', clean_text(text))
    return [token for token in tokens if token not in ENGLISH_STOPWORDS and not token.isdigit()]


def sanitize_input_record(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
    """
    Validates and standardizes a single incoming raw dictionary from form or CSV row.
    Returns (cleaned_dict, error_list).
    """
    errors = []
    
    # Mandatory field: inquiry text
    inquiry_text = str(raw.get("inquiry_text", raw.get("inquiry", raw.get("message", "")))).strip()
    if not inquiry_text or len(inquiry_text) < 10:
        errors.append("Inquiry text is required and must be at least 10 characters long.")

    # Name
    name = str(raw.get("name", raw.get("contact_name", "Prospective Client"))).strip()
    if not name:
        name = "Prospective Client"

    # Email
    email = str(raw.get("email", raw.get("contact_email", "unspecified@example.com"))).strip()
    if "@" not in email or "." not in email:
        errors.append(f"Invalid email format: '{email}'.")

    # Optional fields with sensible defaults
    company = str(raw.get("company", raw.get("organization", "Unspecified"))).strip() or "Unspecified"
    role = str(raw.get("role", raw.get("title", "Unknown"))).strip() or "Unknown"
    budget_tier = str(raw.get("budget_tier", raw.get("budget", "Unspecified / Flexible"))).strip() or "Unspecified / Flexible"
    timeline = str(raw.get("timeline", raw.get("urgency", "Exploratory / No fixed date"))).strip() or "Exploratory / No fixed date"
    company_size = str(raw.get("company_size", "Not Specified")).strip() or "Not Specified"
    source_channel = str(raw.get("source_channel", raw.get("channel", "Website Form"))).strip() or "Website Form"

    cleaned = {
        "name": name,
        "email": email,
        "company": company,
        "role": role,
        "inquiry_text": inquiry_text,
        "budget_tier": budget_tier,
        "timeline": timeline,
        "company_size": company_size,
        "source_channel": source_channel,
    }
    return cleaned, errors
