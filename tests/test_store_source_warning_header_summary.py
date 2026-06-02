from graph.store import summarize_source_warning_headers


def test_warning_summary_extracts_codes_and_invalid_samples():
    summary = summarize_source_warning_headers(
        [
            {"source_id": "a", "Warning": '110 CDN "Response is stale", 214 Proxy "Transformed"'},
            {"source_id": "b", "metadata": {"response_headers": {"WARNING": '112 Cache "Disconn"'}}},
            {"source_id": "c", "headers": {"warning": "not-a-warning"}},
            {"source_id": "d"},
        ],
        sample_limit=1,
    )

    assert summary["sources_with_warning"] == 3
    assert summary["sources_missing_warning"] == 1
    assert summary["warning_code_counts"] == {"110": 1, "112": 1, "214": 1}
    assert summary["stale_warning_count"] == 1
    assert summary["transformation_warning_count"] == 1
    assert summary["invalid_warning_samples"] == [{"source_id": "c", "value": "not-a-warning"}]
