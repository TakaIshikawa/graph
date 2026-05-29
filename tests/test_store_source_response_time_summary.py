from graph.store import summarize_source_response_times


def test_summarizes_top_level_and_metadata_timings_with_buckets():
    summary = summarize_source_response_times(
        [
            {"source_id": "fast", "response_time_ms": "100"},
            {"source_id": "moderate", "metadata": {"elapsed_ms": 250}},
            {"source_id": "slow", "duration_ms": 1200},
            {"source_id": "very", "metadata": {"fetch_duration_ms": "6000"}},
            {"source_id": "bad", "latency_ms": "n/a"},
        ]
    )

    assert summary["total_sources"] == 5
    assert summary["sources_with_timing"] == 4
    assert summary["min_ms"] == 100.0
    assert summary["max_ms"] == 6000.0
    assert summary["average_ms"] == 1887.5
    assert summary["bucket_counts"] == {"fast": 1, "moderate": 1, "slow": 1, "very_slow": 1}
    assert summary["slow_source_samples"] == [
        {"source_id": "very", "timing_key": "fetch_duration_ms", "response_time_ms": 6000.0},
        {"source_id": "slow", "timing_key": "duration_ms", "response_time_ms": 1200.0},
    ]
