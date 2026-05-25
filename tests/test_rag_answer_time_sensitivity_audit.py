from __future__ import annotations

from graph.rag.answer_time_sensitivity_audit import audit_answer_time_sensitivity


def test_time_sensitivity_flags_current_claim_without_evidence_dates():
    result = audit_answer_time_sensitivity(
        "The policy currently allows remote work.",
        [{"id": "e1", "text": "Remote work is described in the handbook."}],
    )

    assert result["findings"][0]["claim_text"] == "The policy currently allows remote work."
    assert result["evidence_date_count"] == 0
    assert result["oldest_detected_date"] is None


def test_time_sensitivity_extracts_dates_and_respects_supported_as_of_claim():
    result = audit_answer_time_sensitivity(
        "As of March 5, 2024, the policy still allows remote work.",
        [{"id": "e1", "text": "Updated March 5, 2024. Remote work remains allowed."}],
    )

    assert result["findings"] == []
    assert result["evidence_date_count"] == 1
    assert result["oldest_detected_date"] == "2024-03-05"
    assert result["newest_detected_date"] == "2024-03-05"


def test_time_sensitivity_extracts_iso_dates_from_metadata():
    result = audit_answer_time_sensitivity(
        "The latest report is available.",
        [{"id": "e1", "metadata": {"published_at": "2024-02-01"}}],
    )

    assert result["findings"] == []
    assert result["evidence_date_count"] == 1
