from __future__ import annotations

import pytest

from graph.rag.query_deletion_right_requirement import detect_query_deletion_right_requirement


def test_detects_deletion_right_cues_and_timing_values():
    result = detect_query_deletion_right_requirement(
        "Show evidence for right to deletion, account deletion, hard delete, "
        "and purge timelines within 30 days or 90 days."
    )

    assert result == {
        "requires_deletion_right": True,
        "cue_categories": ["right_to_deletion", "account_deletion", "purge_timeline", "hard_delete"],
        "timing_values": ["30 days", "90 days"],
    }


def test_detects_erasure_request_soft_delete_and_retention_after_deletion():
    result = detect_query_deletion_right_requirement(
        "How do erasure request workflows handle soft delete and retention after deletion for 7 years?"
    )

    assert result["requires_deletion_right"] is True
    assert result["cue_categories"] == ["data_deletion_request", "soft_delete", "retention_after_deletion"]
    assert result["timing_values"] == ["7 years"]


def test_generic_retention_period_without_deletion_wording_does_not_match():
    assert detect_query_deletion_right_requirement("Compare retention periods for records kept for 90 days.") == {
        "requires_deletion_right": False,
        "cue_categories": [],
        "timing_values": [],
    }


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        detect_query_deletion_right_requirement("  ")
