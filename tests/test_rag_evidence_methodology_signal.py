from __future__ import annotations

from graph.rag.evidence_methodology_signal import extract_evidence_methodology_signals


def test_accepts_strings_and_dicts():
    result = extract_evidence_methodology_signals(["A randomized trial was run.", {"content": "Survey results followed."}])

    assert result["items"][0]["signals"] == ["randomized_trial"]
    assert result["items"][1]["signals"] == ["survey"]


def test_classifies_multiple_methodology_types():
    result = extract_evidence_methodology_signals([{"text": "A benchmark, case study, and simulation."}, {"text": "Systematic review with interviews."}])

    assert result["methodology_types"] == ["benchmark", "case_study", "interview", "simulation", "systematic_review"]


def test_empty_signals_for_no_methodology_language():
    result = extract_evidence_methodology_signals(["Plain product copy."])

    assert result["items"][0]["signals"] == []
    assert result["aggregate_counts"] == {}
