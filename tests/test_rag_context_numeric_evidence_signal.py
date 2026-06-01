from __future__ import annotations

from graph.rag import analyze_context_numeric_evidence_signals


def test_context_numeric_evidence_signal_counts_signal_types_and_density():
    report = analyze_context_numeric_evidence_signals(
        [
            {"id": "c1", "content": "The trial had n=120 and 42% response with p < 0.05."},
            {"id": "c2", "text": "Costs ranged 10-20 USD and revenue was $1,200."},
            {"id": "c3", "text": "No numbers here."},
        ]
    )

    assert report["context_count"] == 3
    assert report["contexts_with_numeric_evidence"] == 2
    assert report["signal_counts"]["sample_size"] == 1
    assert report["signal_counts"]["percent"] == 1
    assert report["signal_counts"]["p_value"] == 1
    assert report["numeric_density"] == 2.0


def test_context_numeric_evidence_signal_handles_empty_contexts():
    report = analyze_context_numeric_evidence_signals([])

    assert report["context_count"] == 0
    assert report["numeric_density"] == 0.0
