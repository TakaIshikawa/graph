from __future__ import annotations

from graph.store.relation_evidence_quality_summary import summarize_relation_evidence_quality


def test_summarize_relation_evidence_quality_counts_missing_weak_and_strong():
    summary = summarize_relation_evidence_quality(
        [
            {"relation_type": "depends_on"},
            {"relation_type": "mentions", "evidence": ["quote"], "confidence": 0.4},
            {"relation_type": "depends_on", "metadata": {"evidence": ["quote", "link"], "confidence": 0.9}},
        ]
    )

    assert summary == {
        "missing_evidence_count": 1,
        "weak_evidence_count": 1,
        "strong_evidence_count": 1,
        "average_evidence_count": 1.0,
        "average_confidence": 0.65,
        "counts_by_relation_type": [
            {"relation_type": "depends_on", "count": 2},
            {"relation_type": "mentions", "count": 1},
        ],
    }


def test_summarize_relation_evidence_quality_ignores_absent_confidence_values():
    summary = summarize_relation_evidence_quality(
        [{"relation_type": "related_to", "evidence": ["one"]}, {"relation_type": "related_to", "evidence": ["two"]}]
    )

    assert summary["average_confidence"] == 0.0
    assert summary["weak_evidence_count"] == 2
