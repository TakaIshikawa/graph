from __future__ import annotations

from graph.rag.query_confidence_threshold import detect_query_confidence_thresholds


def test_detects_numeric_threshold_and_confidence_terms():
    result = detect_query_confidence_thresholds("Answer only if you are sure and at least 95% confident.")

    assert result["requires_confidence_handling"] is True
    assert result["threshold_values"] == ["95%"]
    assert "sure" in result["confidence_terms"]


def test_detects_uncertainty_flagging_without_threshold():
    result = detect_query_confidence_thresholds("Flag uncertainty and note evidence strength.")

    assert result["requires_confidence_handling"] is True
    assert result["threshold_values"] == []
    assert "flag_uncertainty" in result["confidence_terms"]


def test_plain_query_has_no_confidence_requirement():
    result = detect_query_confidence_thresholds("Summarize the onboarding process.")

    assert result["requires_confidence_handling"] is False
    assert result["matched_phrases"] == []
