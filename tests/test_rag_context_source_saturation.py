from __future__ import annotations

from graph.rag.context_source_saturation import analyze_context_source_saturation


def test_computes_repetition_metrics():
    result = analyze_context_source_saturation(
        [
            {"id": "c1", "source_id": "s1", "url": "https://a.test/1", "author": "A", "title": "One"},
            {"id": "c2", "source_id": "s1", "url": "https://a.test/2", "author": "A", "title": "Two"},
            {"id": "c3", "source_id": "s2", "url": "https://b.test/1", "author": "B", "title": "Three"},
        ]
    )

    assert result["context_count"] == 3
    assert result["metrics"]["source_id"]["dominant_value"] == "s1"
    assert result["metrics"]["domain"]["unique_count"] == 2


def test_flags_dominant_source_saturation():
    result = analyze_context_source_saturation(
        [
            {"source_id": "s1", "url": "https://a.test/1"},
            {"source_id": "s1", "url": "https://a.test/2"},
            {"source_id": "s1", "url": "https://a.test/3"},
            {"source_id": "s2", "url": "https://b.test/1"},
        ],
        dominance_threshold=0.5,
    )

    assert result["metrics"]["source_id"]["saturated"] is True
    assert "dominant_source_id:s1" in result["warnings"]


def test_handles_missing_metadata():
    result = analyze_context_source_saturation([{"id": "c1"}, {"id": "c2"}])

    assert result["metrics"]["source_id"]["dominant_value"] == "missing"
    assert result["warnings"] == []
