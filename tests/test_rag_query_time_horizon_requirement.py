from __future__ import annotations

from graph.rag.query_time_horizon_requirement import detect_query_time_horizon_requirement


def test_detects_explicit_bounded_date_range():
    result = detect_query_time_horizon_requirement("Compare adoption from 2020 to 2023.")

    assert result["requires_time_horizon"] is True
    assert "bounded_range" in result["horizon_types"]
    assert result["date_ranges"] == [{"start": "2020", "end": "2023", "text": "from 2020 to 2023"}]


def test_detects_forecast_wording():
    result = detect_query_time_horizon_requirement("Forecast demand for the next 2 years.")

    assert "forecast" in result["horizon_types"]
    assert set(result["matched_terms"]["forecast"]) == {"forecast", "next"}
    assert result["confidence"] > 0


def test_detects_historical_wording():
    result = detect_query_time_horizon_requirement("Show historical churn over time since 2019.")

    assert "historical" in result["horizon_types"]
    assert "historical" in result["matched_terms"]["historical"]
    assert "since" in result["matched_terms"]["historical"]


def test_no_horizon_query_is_unflagged():
    result = detect_query_time_horizon_requirement("Explain how vector search works.")

    assert result["requires_time_horizon"] is False
    assert result["horizon_types"] == []
    assert result["confidence"] == 0.0
