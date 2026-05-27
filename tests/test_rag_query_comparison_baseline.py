from __future__ import annotations

from graph.rag.query_comparison_baseline import detect_query_comparison_baselines


def test_detects_versus_and_baseline_terms():
    result = detect_query_comparison_baselines("Compare onboarding conversion versus last quarter baseline.")

    assert result["requires_comparison"] is True
    assert "versus" in result["comparison_terms"]
    assert "conversion" in result["baseline_terms"]


def test_detects_before_after_and_compared_with():
    result = detect_query_comparison_baselines("Show retention before and after launch compared with baseline.")

    assert result["requires_comparison"] is True
    assert "before_and_after" in result["comparison_terms"]
    assert "compared_with" in result["comparison_terms"]


def test_avoids_substring_false_positives():
    result = detect_query_comparison_baselines("Explain version history and conversion tracking.")

    assert result["requires_comparison"] is False
    assert result["matched_phrases"] == []
