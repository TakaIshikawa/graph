from __future__ import annotations

from graph.rag.query_time_sensitivity import detect_query_time_sensitivity


def test_query_time_sensitivity_flags_freshness_terms():
    result = detect_query_time_sensitivity("latest price forecast and release schedule")

    assert result["requires_fresh_context"] is True
    assert result["matched_terms"] == ["latest", "forecast", "price", "schedule", "release"]
    assert result["suggested_recency_days"] == 30


def test_query_time_sensitivity_leaves_stable_historical_queries_unflagged():
    result = detect_query_time_sensitivity("What caused the Roman aqueduct system to expand?")

    assert result["requires_fresh_context"] is False
    assert result["matched_terms"] == []
    assert result["suggested_recency_days"] is None
