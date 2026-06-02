from __future__ import annotations

from graph.rag.query_certificate_lifecycle_requirement import detect_query_certificate_lifecycle_requirement


def test_detects_certificate_lifecycle_categories():
    result = detect_query_certificate_lifecycle_requirement(
        "Need certificate renewal, certificate expiration monitoring, ACME automation, CA rotation, and mTLS certificate lifecycle evidence."
    )

    assert result == {
        "requires_certificate_lifecycle": True,
        "cue_categories": ["renewal", "expiration_monitoring", "automation", "ca_rotation", "mtls_lifecycle"],
    }


def test_overlapping_certificate_phrases_are_deterministic():
    result = detect_query_certificate_lifecycle_requirement(
        "Automated certificate renewal and renew certificates before expiry."
    )

    assert result["cue_categories"] == ["renewal", "automation"]


def test_general_tls_certificate_question_does_not_match():
    assert detect_query_certificate_lifecycle_requirement("Does the browser trust this TLS certificate?") == {
        "requires_certificate_lifecycle": False,
        "cue_categories": [],
    }
