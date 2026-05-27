from __future__ import annotations

from graph.rag.query_precision_requirement import detect_query_precision_requirement


def test_query_precision_requirement_detects_exact_numeric_requests():
    result = detect_query_precision_requirement("Give the exact value to 2 decimal places")

    assert result["precision_level"] == "exact"
    assert result["numeric_precision_requested"] is True


def test_query_precision_requirement_detects_approximate_and_range_requests():
    assert detect_query_precision_requirement("roughly estimate the cost")["precision_level"] == "approximate"
    assert detect_query_precision_requirement("show the expected range")["precision_level"] == "range"


def test_query_precision_requirement_uses_precedence_for_mixed_cues():
    result = detect_query_precision_requirement("approximately exact to nearest 10")

    assert result["precision_level"] == "exact"
    assert result["matched_cues"] == ["approximately", "exact", "nearest"]
