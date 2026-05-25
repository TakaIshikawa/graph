from __future__ import annotations

from graph.rag.context_gap_prioritizer import prioritize_context_gaps


def test_context_gap_prioritizer_ranks_high_severity_above_low_impact():
    result = prioritize_context_gaps(
        [
            {"gap_type": "definition", "severity": "low", "answer_impact": "low"},
            {"gap_type": "missing_date", "severity": "high", "answer_impact": "high"},
        ],
        "latest policy",
    )

    assert result["prioritized_gaps"][0]["gap_type"] == "missing_date"
    assert result["prioritized_gaps"][0]["suggested_retrieval_action"] == "retrieve current dated sources"


def test_context_gap_prioritizer_preserves_unknown_gap_shapes():
    result = prioritize_context_gaps([{"message": "unclear"}])

    assert result["prioritized_gaps"][0]["message"] == "unclear"
    assert result["prioritized_gaps"][0]["gap_type"] == "unknown"
    assert result["prioritized_gaps"][0]["priority"] == "medium"
