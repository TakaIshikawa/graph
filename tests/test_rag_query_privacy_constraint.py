from __future__ import annotations

from graph.rag import detect_query_privacy_constraints


def test_query_privacy_constraint_detects_local_redaction_and_external_service_limits():
    report = detect_query_privacy_constraints("Keep this local only, redact personal info, and avoid external APIs.")

    assert report["has_privacy_constraint"] is True
    assert report["local_only"] is True
    assert report["redact_pii"] is True
    assert report["avoid_external_services"] is True
    assert report["public_ok"] is False
    assert report["confidence"] > 0.5


def test_query_privacy_constraint_distinguishes_public_ok():
    report = detect_query_privacy_constraints("This can share publicly.")

    assert report["has_privacy_constraint"] is False
    assert report["public_ok"] is True
