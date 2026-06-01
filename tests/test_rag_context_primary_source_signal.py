from __future__ import annotations

from graph.rag.context_primary_source_signal import analyze_context_primary_source_signals


def test_context_primary_source_signal_classifies_sources():
    summary = analyze_context_primary_source_signals(
        [
            {"id": "official", "url": "https://agency.gov/rule", "title": "Rule"},
            {"id": "journal", "metadata": {"source_type": "journal"}},
            {"id": "blog", "source_type": "blog summary"},
            {"id": "missing"},
        ]
    )

    assert summary["classification_counts"] == {"primary_source": 2, "secondary_source": 1, "unknown": 1}
    assert [row["classification"] for row in summary["contexts"]] == ["primary_source", "primary_source", "secondary_source", "unknown"]
