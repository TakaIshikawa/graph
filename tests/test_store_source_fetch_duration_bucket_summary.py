from __future__ import annotations

from graph.store.source_fetch_duration_bucket_summary import summarize_source_fetch_duration_buckets


def test_source_fetch_duration_buckets_parse_values_and_timing_containers():
    summary = summarize_source_fetch_duration_buckets(
        [
            {"source_id": "a", "fetch_duration_ms": "99"},
            {"source_id": "b", "metadata": {"response_time_ms": 100}},
            {"source_id": "c", "timing": {"elapsed_ms": "500"}},
            {"source_id": "d", "metadata": {"timing": {"duration_ms": 1000}}},
            {"source_id": "e", "response_time_ms": 5000},
            {"source_id": "f", "duration_ms": "-1"},
            {"source_id": "g", "elapsed_ms": "slow"},
            {"source_id": "h"},
        ],
        sample_limit=1,
    )

    assert summary["total_sources"] == 8
    assert summary["sources_with_fetch_duration"] == 5
    assert summary["invalid_duration_count"] == 2
    assert summary["bucket_counts"] == {
        "<100ms": 1,
        "100-499ms": 1,
        "500-999ms": 1,
        "1-4.999s": 1,
        ">=5s": 1,
    }
    assert summary["slow_source_count"] == 1
    assert summary["slow_samples"] == [{"source_id": "e", "duration_ms": 5000.0, "bucket": ">=5s"}]


def test_source_fetch_duration_slow_samples_are_sorted_and_limited():
    summary = summarize_source_fetch_duration_buckets(
        [
            {"source_id": "b", "duration_ms": 6000},
            {"source_id": "a", "duration_ms": 7000},
        ],
        sample_limit=1,
    )

    assert summary["slow_samples"] == [{"source_id": "a", "duration_ms": 7000.0, "bucket": ">=5s"}]
