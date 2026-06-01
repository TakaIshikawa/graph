from __future__ import annotations

from graph.rag import analyze_result_conflict_signals


def test_result_conflict_signal_classifies_conflict_severity():
    report = analyze_result_conflict_signals(
        [
            {"id": "r1", "title": "Retraction notice", "snippet": "The paper was retracted."},
            {"id": "r2", "content": "The evidence is controversial and inconsistent."},
        ]
    )

    assert report["result_count"] == 2
    assert report["results_with_conflict_signal"] == 2
    assert report["severity_counts"] == {"high": 1, "low": 2}
    assert report["samples"][0]["result_id"] == "r1"
    assert report["samples"][0]["severity"] == "high"
