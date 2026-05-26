from __future__ import annotations

from graph.rag import detect_query_comparison_axes


def test_comparison_axis_detector_detects_multiple_axes_case_insensitively():
    result = detect_query_comparison_axes("Compare COST, latency, and privacy for these vendors")

    assert [axis["axis"] for axis in result["axes"]] == ["cost", "latency", "privacy"]
    assert result["confidence"] == 0.9


def test_comparison_axis_detector_returns_empty_axes_for_non_comparison_query():
    assert detect_query_comparison_axes("Summarize the deployment guide") == {"axes": [], "confidence": 0.1}
