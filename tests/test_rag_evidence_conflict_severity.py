from __future__ import annotations

from graph.rag.evidence_conflict_severity import classify_evidence_conflict_severity, score_evidence_conflict_severity


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


def test_classifies_direct_contradiction_and_numeric_disagreement_as_high():
    report = classify_evidence_conflict_severity(["Direct contradiction with numeric disagreement: 20 vs 80."])

    assert report["severity_counts"]["high"] == 1
    assert "direct_contradiction" in report["classifications"][0]["reasons"]


def test_classifies_date_or_source_disagreement_as_medium():
    report = classify_evidence_conflict_severity([{"text": "Date mismatch between sources."}, "Source disagreement only."])

    assert report["severity_counts"]["medium"] == 2


def test_classifies_wording_only_difference_as_low_with_reason():
    report = classify_evidence_conflict_severity(["Minor wording difference in summaries."])

    assert report["severity_counts"]["low"] == 1
    assert report["classifications"][0]["reasons"] == ["minor_wording_difference"]
