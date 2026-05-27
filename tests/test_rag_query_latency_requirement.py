from __future__ import annotations

from graph.rag.query_latency_requirement import detect_query_latency_requirement


def test_detect_query_latency_requirement_normalizes_time_limits_to_seconds():
    result = detect_query_latency_requirement("Need results within 200ms, under 2 seconds, below 3 minutes, and in 1 hour.")

    assert [row["seconds"] for row in result["numeric_limits"]] == [0.2, 2.0, 180.0, 3600.0]
    assert result["latency_class"] == "fast"


def test_detect_query_latency_requirement_maps_real_time_and_batch_classes():
    real_time = detect_query_latency_requirement("Use real-time sources for a low latency answer.")
    batch = detect_query_latency_requirement("This can run as a batch background job.")

    assert real_time["latency_class"] == "real_time"
    assert batch["latency_class"] == "batch"


def test_detect_query_latency_requirement_returns_unconstrained_without_latency_language():
    assert detect_query_latency_requirement("Summarize evidence quality by source type.") == {
        "requires_latency_awareness": False,
        "latency_class": "unconstrained",
        "numeric_limits": [],
        "cues": [],
    }
