from __future__ import annotations

from graph.rag.query_sensitivity_threshold import detect_query_sensitivity_threshold


def test_detects_lower_upper_range_percentile_and_materiality_thresholds():
    report = detect_query_sensitivity_threshold("Alert at least 10%, no more than 5 days, above p95, within tolerance, and material change.")

    assert report["has_threshold"] is True
    assert report["threshold_types"] == ["lower_bound", "materiality", "percentile", "range", "upper_bound"]
    assert "10%" in report["numeric_thresholds"]
    assert "5 days" in report["numeric_thresholds"]
    assert "p95" in report["numeric_thresholds"]


def test_detects_under_percent_threshold():
    report = detect_query_sensitivity_threshold("Keep error rate under 5%.")

    assert report["threshold_types"] == ["upper_bound"]
    assert report["numeric_thresholds"] == ["5%"]


def test_non_threshold_numbers_are_ignored():
    report = detect_query_sensitivity_threshold("Summarize 5 incidents from 2024.")

    assert report == {"has_threshold": False, "threshold_types": [], "matched_cues": [], "numeric_thresholds": []}
