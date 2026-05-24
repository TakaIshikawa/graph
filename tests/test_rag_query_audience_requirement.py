from __future__ import annotations

from graph.rag.query_audience_requirement import detect_query_audience_requirement


def test_normalizes_beginner_audience():
    result = detect_query_audience_requirement("Explain this in plain English for beginners.")

    assert result["requires_audience_adaptation"] is True
    assert result["audiences"] == ["beginner"]
    assert result["requirements"][0]["suggested_explanation_depth"] == "introductory"


def test_supports_multiple_audiences():
    result = detect_query_audience_requirement("Summarize for executives and engineers.")

    assert result["audiences"] == ["executive", "engineer"]
    assert "executives" in result["matched_phrases"]
    assert "engineers" in result["matched_phrases"]


def test_detects_clinician_and_policymaker():
    result = detect_query_audience_requirement("Write this for clinicians and policymakers.")

    assert result["audiences"] == ["clinician", "policymaker"]


def test_neutral_query_is_unflagged():
    result = detect_query_audience_requirement("What are the causes of latency?")

    assert result["requires_audience_adaptation"] is False
    assert result["requirements"] == []
