from graph.store.source_server_timing_summary import summarize_source_server_timings


def test_server_timing_summary_parses_metrics_durations_and_descriptions():
    summary = summarize_source_server_timings(
        [
            {"source_id": "a", "Server-Timing": 'dns;dur=12.5;desc="DNS lookup", app;dur=640'},
            {"source_id": "b", "metadata": {"response_headers": {"server-timing": "app;dur=bad, cache;desc=hit"}}},
            {"source_id": "c", "headers": {"SERVER_TIMING": "edge"}},
            {"source_id": "d"},
        ],
        sample_limit=3,
    )

    assert summary["sources_with_server_timing"] == 3
    assert summary["metric_counts"] == {"app": 2, "cache": 1, "dns": 1, "edge": 1}
    assert summary["duration_buckets"] == {"lt_100ms": 1, "500_999ms": 1}
    assert summary["missing_duration_count"] == 2
    assert summary["invalid_duration_count"] == 1
    assert summary["description_count"] == 2
    assert summary["missing_server_timing_count"] == 1
    assert summary["samples"] == [
        {"source_id": "a", "metric": "dns", "value": 'dns;dur=12.5;desc="DNS lookup"', "dur": "12.5", "desc": "DNS lookup"},
        {"source_id": "a", "metric": "app", "value": "app;dur=640", "dur": "640"},
        {"source_id": "b", "metric": "app", "value": "app;dur=bad", "dur": "bad"},
    ]
