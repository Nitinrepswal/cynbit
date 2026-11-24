"""
Unit tests for Cynbit LeadIQ preprocessing routines.
"""

import pytest
from src.preprocessing import clean_text, tokenize_and_filter, sanitize_input_record


def test_clean_text_normalizes_case_and_whitespace():
    raw = "  Hello   WORLD!   Need An AI Engine.  "
    cleaned = clean_text(raw)
    assert cleaned == "hello world need an ai engine."


def test_clean_text_strips_html_and_urls():
    raw = "<p>Check out our site at https://example.com/demo?ref=123 &amp; build an app</p>"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "<p>" not in cleaned
    assert "&amp;" not in cleaned
    assert "build an app" in cleaned


def test_clean_text_removes_emails():
    raw = "Contact our lead at test.user@enterprise.org regarding the pipeline"
    cleaned = clean_text(raw)
    assert "@" not in cleaned
    assert "regarding the pipeline" in cleaned


def test_tokenize_and_filter_removes_stopwords():
    text = "We are very interested in machine learning and artificial intelligence"
    tokens = tokenize_and_filter(text)
    assert "machine" in tokens
    assert "learning" in tokens
    assert "artificial" in tokens
    assert "intelligence" in tokens
    assert "are" not in tokens
    assert "and" not in tokens
    assert "in" not in tokens


def test_sanitize_input_record_valid():
    raw = {
        "name": "Jane Doe",
        "email": "jane@corp.com",
        "inquiry_text": "We want to build a real-time web portal for IoT tracking."
    }
    cleaned, errors = sanitize_input_record(raw)
    assert len(errors) == 0
    assert cleaned["name"] == "Jane Doe"
    assert cleaned["email"] == "jane@corp.com"
    assert cleaned["company"] == "Unspecified"


def test_sanitize_input_record_invalid_short_inquiry_and_bad_email():
    raw = {
        "name": "Bob",
        "email": "invalid-email-address",
        "inquiry_text": "Hi"
    }
    cleaned, errors = sanitize_input_record(raw)
    assert len(errors) >= 2
    assert any("at least 10 characters" in e for e in errors)
    assert any("Invalid email format" in e for e in errors)
