from __future__ import annotations

from graph.rag.context_methodology_signal import analyze_context_methodology_signals


def test_context_methodology_signal_counts_contexts_and_samples():
    summary = analyze_context_methodology_signals(
        [
            {"id": "a", "title": "Study", "content": "The dataset includes a sample size of 200."},
            {"id": "b", "snippet": "A survey and interview protocol described one limitation."},
            {"id": "c", "text": "Background only."},
        ]
    )

    assert summary["contexts_with_methodology"] == 2
    assert summary["contexts_without_methodology"] == 1
    assert summary["signal_counts"]["dataset"] == 1
    assert summary["signal_counts"]["survey"] == 1
    assert summary["samples"][0] == {"context_id": "a", "title": "Study", "signals": ["sample size", "dataset"]}
