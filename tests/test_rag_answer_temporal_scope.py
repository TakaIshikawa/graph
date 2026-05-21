from __future__ import annotations

from graph.rag.answer_temporal_scope import audit_answer_temporal_scope


def test_answer_temporal_scope_identifies_claims_and_bounds():
    report = audit_answer_temporal_scope(
        "The policy changed in 2024 and was revised on 2025-06-01.",
        "latest policy after 2023",
        [{"id": "a", "date": "2024-01-10"}, {"id": "b", "metadata": {"published_at": "2025-05-01"}}],
    )

    assert [claim["text"] for claim in report["date_claims"]] == ["2024", "2025-06-01"]
    assert report["evidence_date_bounds"]["oldest_date"] == "2024-01-10"
    assert report["evidence_date_bounds"]["newest_date"] == "2025-05-01"
    assert "answer_date_after_evidence" in report["warnings"]
    assert report["query_expectations"]["requires_current"] is True


def test_answer_temporal_scope_flags_query_scope_and_missing_dates():
    report = audit_answer_temporal_scope("Use the 2020 guidance.", "after 2022", [])

    assert "no_dated_evidence" in report["warnings"]
    assert "answer_date_before_query_scope" in report["warnings"]
    assert report["status"] == "warning"
