from graph.store.source_server_timing_summary import summarize_source_server_timings


def test_source_server_timing_summary_reads_direct_and_nested_headers():
    summary = summarize_source_server_timings(
        [
            {"source_id": "direct", "Server-Timing": 'Cache;desc="Hit";dur=23'},
            {"source_id": "nested", "response_headers": {"server_timing": "db;dur=101"}},
            {"source_id": "metadata", "metadata": {"response_headers": {"SERVER-TIMING": "app"}}},
            {"source_id": "missing"},
        ],
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_server_timing"] == 3
    assert summary["missing_server_timing_count"] == 1
    assert summary["metric_counts"] == {"app": 1, "cache": 1, "db": 1}


def test_source_server_timing_summary_parses_multiple_metrics_and_sorts_samples():
    summary = summarize_source_server_timings(
        [
            {"source_id": "z", "Server-Timing": "zeta;dur=1, alpha;desc=first"},
            {"source_id": "a", "Server-Timing": "CACHE;dur=2, cache;dur=3"},
        ],
        sample_limit=3,
    )

    assert summary["metric_counts"] == {"alpha": 1, "cache": 2, "zeta": 1}
    assert summary["rows"] == [
        {"metric": "alpha", "count": 1, "source_ids": ["z"], "examples": ["alpha;desc=first"]},
        {"metric": "cache", "count": 2, "source_ids": ["a"], "examples": ["CACHE;dur=2", "cache;dur=3"]},
        {"metric": "zeta", "count": 1, "source_ids": ["z"], "examples": ["zeta;dur=1"]},
    ]
    assert [row["source_id"] for row in summary["samples"]] == ["a", "a", "z"]
    assert [row["metric"] for row in summary["samples"]] == ["cache", "cache", "zeta"]


def test_source_server_timing_summary_bounds_row_source_ids_and_examples():
    summary = summarize_source_server_timings(
        [
            {"source_id": "a", "Server-Timing": "cache;dur=1"},
            {"source_id": "b", "Server-Timing": "cache;dur=2"},
            {"source_id": "c", "Server-Timing": "cache;dur=3"},
        ],
        sample_limit=2,
    )

    assert summary["rows"] == [
        {"metric": "cache", "count": 3, "source_ids": ["a", "b"], "examples": ["cache;dur=1", "cache;dur=2"]}
    ]


def test_source_server_timing_summary_ignores_blank_and_malformed_segments():
    summary = summarize_source_server_timings(
        [
            {"source_id": "bad", "Server-Timing": ', ;dur=1, =oops, "quoted";dur=4'},
            {"source_id": "good", "Server-Timing": "edge;dur=bad, cache;desc=\"Hit, from edge\""},
        ]
    )

    assert summary["sources_with_server_timing"] == 1
    assert summary["missing_server_timing_count"] == 1
    assert summary["metric_counts"] == {"cache": 1, "edge": 1}
    assert summary["invalid_duration_count"] == 1
    assert summary["description_count"] == 1
