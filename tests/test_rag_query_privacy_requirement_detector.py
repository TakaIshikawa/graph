from __future__ import annotations

from graph.rag.query_privacy_requirement_detector import detect_query_privacy_requirements


def test_query_privacy_requirement_flags_sensitive_cues():
    report = detect_query_privacy_requirements("Redact PII, API keys, medical records, financial data, and legal notes.")

    assert report["requires_privacy_handling"] is True
    assert report["sensitive_topics"] == ["pii", "credentials", "medical", "financial", "legal", "anonymization"]
    assert report["confidence"] == 0.85


def test_query_privacy_requirement_detects_anonymization_and_private_notes():
    report = detect_query_privacy_requirements("Anonymize private notes before using them.")

    assert report["sensitive_topics"] == ["private_notes", "anonymization"]
    assert [cue["text"] for cue in report["matched_cues"]] == ["private notes", "Anonymize"]


def test_query_privacy_requirement_returns_stable_unmatched_guidance():
    report = detect_query_privacy_requirements("Find public documentation for the API.")

    assert report["requires_privacy_handling"] is False
    assert report["matched_cues"] == []
    assert report["guidance"] == "No special privacy handling detected."
