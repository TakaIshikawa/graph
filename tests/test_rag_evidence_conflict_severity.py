from __future__ import annotations

from graph.rag.evidence_conflict_severity import score_evidence_conflict_severity


def test_scores_high_numeric_and_date_conflicts_with_sources():
    report = score_evidence_conflict_severity(
        [
            {"id": "a", "source_id": "s1", "text": "Widget adoption increased to 40% in 2024."},
            {"id": "b", "source_id": "s2", "text": "Widget adoption decreased to 20% in 2025."},
        ]
    )

    assert report["max_severity"] == "high"
    assert report["severity_counts"]["high"] == 1
    assert report["conflicts"][0]["source_ids"] == ["s1", "s2"]
    assert "numeric_mismatch" in report["conflicts"][0]["reasons"]


def test_scores_medium_for_opposing_polarity_without_numeric_mismatch():
    report = score_evidence_conflict_severity(
        [
            {"id": "a", "text": "Policy support is available for teams."},
            {"id": "b", "text": "Policy support is unavailable for teams."},
        ]
    )

    assert report["max_severity"] == "medium"
