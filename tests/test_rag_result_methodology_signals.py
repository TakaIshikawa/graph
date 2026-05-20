from __future__ import annotations

from graph.rag.result_methodology_signals import extract_result_methodology_signals


def test_result_methodology_signals_detects_strong_evidence():
    rows = extract_result_methodology_signals(
        [{"id": "study", "content": "Survey of 200 participants using a benchmark dataset measured accuracy."}]
    )

    assert rows[0]["signals"] == ["sample_size", "dataset", "measurement", "benchmark", "survey"]
    assert rows[0]["warnings"] == []
    assert rows[0]["methodology_score"] == 1.0


def test_result_methodology_signals_warns_on_weak_commentary():
    rows = extract_result_methodology_signals([{"id": "opinion", "content": "This feels better than alternatives."}])

    assert rows[0]["signals"] == []
    assert rows[0]["warnings"] == ["missing_methodology_signals"]
    assert rows[0]["methodology_score"] == 0.0


def test_result_methodology_signals_uses_metadata_only_signals():
    rows = extract_result_methodology_signals([{"id": "meta", "metadata": {"method": "test set evaluation"}}])

    assert rows[0]["signals"] == ["benchmark"]
