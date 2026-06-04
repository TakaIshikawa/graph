from __future__ import annotations

from graph.rag.query_rto_rpo_requirement import detect_query_rto_rpo_requirements


def test_rto_rpo_requirement_rejects_unrelated_recovery_language():
    result = detect_query_rto_rpo_requirements("Recover the deleted paragraph and summarize it.")

    assert result["has_rto_rpo_requirements"] is False
    assert result["requirements"] == []


def test_rto_and_rpo_terms_include_matched_text_and_severity():
    result = detect_query_rto_rpo_requirements("Need RTO under 4 hours and recovery point objective evidence.")

    assert result["has_rto_rpo_requirements"] is True
    assert result["requirements"] == [
        {"category": "rpo", "matched_text": "recovery point objective", "severity": "high"},
        {"category": "rto", "matched_text": "RTO", "severity": "high"},
    ]


def test_rto_rpo_requirement_rows_are_sorted_by_category():
    result = detect_query_rto_rpo_requirements(
        "Show restore validation, acceptable data loss, maximum tolerable downtime, and restore time."
    )

    assert [row["category"] for row in result["requirements"]] == [
        "data_loss_tolerance",
        "downtime_tolerance",
        "restore_validation",
        "rto",
    ]
