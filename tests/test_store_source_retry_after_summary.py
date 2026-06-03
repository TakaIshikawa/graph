from graph.store import summarize_source_retry_after_headers


def test_retry_after_summary_classifies_numeric_dates_invalid_empty_and_missing():
    summary = summarize_source_retry_after_headers(
        [
            {"id": "n1", "Retry-After": "120"},
            {"id": "d", "metadata": {"retry_after": "Tue, 15 Nov 1994 08:12:31 GMT"}},
            {"id": "bad", "headers": {"retry-after": "soon"}},
            {"id": "n2", "metadata": {"response_headers": {"Retry_After": "30"}}},
            {"id": "e", "response_headers": {"Retry-After": ""}},
            {"id": "m"},
        ]
    )

    assert summary["total_sources"] == 6
    assert summary["sources_with_retry_after"] == 4
    assert summary["missing_header_count"] == 1
    assert summary["empty_value_count"] == 1
    assert summary["numeric_delay_count"] == 2
    assert summary["http_date_count"] == 1
    assert summary["invalid_value_count"] == 1
    assert summary["max_delay_seconds"] == 120
    assert [sample["source_id"] for sample in summary["samples"]] == ["bad", "d", "n1", "n2"]


def test_retry_after_summary_respects_sample_limit():
    summary = summarize_source_retry_after_headers(
        [{"id": "b", "Retry-After": "1"}, {"id": "a", "Retry-After": "2"}],
        sample_limit=1,
    )

    assert summary["numeric_delay_count"] == 2
    assert summary["samples"] == [{"source_id": "b", "retry_after": "1", "kind": "numeric_delay"}]
