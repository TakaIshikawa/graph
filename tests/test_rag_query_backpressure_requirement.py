from __future__ import annotations

from graph.rag.query_backpressure_requirement import detect_query_backpressure_requirement


def test_detects_backpressure_categories():
    result = detect_query_backpressure_requirement(
        "Need backpressure when queue saturation occurs, with load shedding, bounded buffers, "
        "throttling pressure, and producer slowdown."
    )

    assert result["has_backpressure_requirement"] is True
    assert result["requirements"] == [
        {"category": "backpressure", "matched_text": "backpressure"},
        {"category": "bounded_buffer", "matched_text": "bounded buffers"},
        {"category": "load_shedding", "matched_text": "load shedding"},
        {"category": "producer_slowdown", "matched_text": "producer slowdown"},
        {"category": "queue_saturation", "matched_text": "queue saturation"},
        {"category": "throttling_pressure", "matched_text": "throttling pressure"},
    ]


def test_ignores_unrelated_performance_wording():
    assert detect_query_backpressure_requirement(
        "Improve performance under heavy traffic and make the dashboard faster."
    ) == {"has_backpressure_requirement": False, "requirements": []}
