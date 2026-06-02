from graph.store import summarize_source_pragma_headers


def test_pragma_summary_counts_no_cache_and_unusual_values():
    summary = summarize_source_pragma_headers(
        [
            {"source_id": "a", "Pragma": "No-Cache"},
            {"source_id": "b", "metadata": {"headers": {"PRAGMA": "custom"}}},
            {"source_id": "c", "response_headers": {"pragma": "debug"}},
            {"source_id": "d"},
        ],
        sample_limit=1,
    )

    assert summary["sources_with_pragma"] == 3
    assert summary["sources_missing_pragma"] == 1
    assert summary["no_cache_count"] == 1
    assert summary["other_value_count"] == 2
    assert summary["top_pragma_values"] == {"custom": 1, "debug": 1, "no-cache": 1}
    assert summary["unusual_pragma_samples"] == [{"source_id": "b", "value": "custom"}]
