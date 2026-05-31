from __future__ import annotations

import pytest

from graph.rag.query_redaction_requirement import detect_query_redaction_requirement


def test_redaction_requirement_rejects_empty_or_non_string_query():
    with pytest.raises(ValueError):
        detect_query_redaction_requirement("")
    with pytest.raises(ValueError):
        detect_query_redaction_requirement(None)  # type: ignore[arg-type]


def test_detects_redaction_verbs_and_sensitive_entities_independently():
    redaction_only = detect_query_redaction_requirement("Please anonymize and mask the retrieved context.")
    entity_only = detect_query_redaction_requirement("Summarize emails, phone numbers, SSN, and API keys.")

    assert redaction_only["requires_redaction"] is True
    assert redaction_only["redaction_terms"] == ["anonymization", "masking"]
    assert redaction_only["sensitive_entity_terms"] == []
    assert entity_only["requires_redaction"] is True
    assert entity_only["redaction_terms"] == []
    assert entity_only["sensitive_entity_terms"] == ["api_keys", "emails", "phone_numbers", "ssn"]


def test_deduplicates_and_sorts_signal_lists():
    report = detect_query_redaction_requirement(
        "Redact emails and email addresses, remove secrets, delete tokens, and de-identify PII."
    )

    assert report["redaction_terms"] == ["de-identification", "redaction", "secret_removal"]
    assert report["sensitive_entity_terms"] == ["emails", "pii", "secrets", "tokens"]
    assert report["rationale"] == "Query includes redaction instructions and sensitive entity terms."


def test_returns_negative_result_for_public_query():
    report = detect_query_redaction_requirement("Find public release notes for the SDK.")

    assert report == {
        "requires_redaction": False,
        "redaction_terms": [],
        "sensitive_entity_terms": [],
        "rationale": "No redaction requirement detected.",
    }
