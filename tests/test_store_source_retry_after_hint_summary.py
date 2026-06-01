from graph.store.source_retry_after_hint_summary import summarize_source_retry_after_hints


def test_retry_after_hint_summary_classifies_seconds_dates_and_invalid_values():
    summary = summarize_source_retry_after_hints(
        [
            {"source_id": "seconds", "metadata": {"retry_after": "120"}},
            {"source_id": "date", "headers": {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}},
            {"source_id": "invalid", "metadata": {"headers": {"retry-after": "soon"}}},
            {"source_id": "missing", "metadata": {"url": "https://example.test"}},
        ]
    )

    assert summary["sources_with_retry_after_hint"] == 3
    assert summary["retry_after_count"] == 3
    assert summary["value_type_counts"] == {"http-date": 1, "invalid": 1, "seconds": 1}
    assert summary["samples"] == [
        {"source_id": "date", "value": "Wed, 21 Oct 2015 07:28:00 GMT", "value_type": "http-date"},
        {"source_id": "invalid", "value": "soon", "value_type": "invalid"},
        {"source_id": "seconds", "value": "120", "value_type": "seconds"},
    ]


def test_retry_after_hint_summary_finds_common_metadata_and_header_casings():
    summary = summarize_source_retry_after_hints(
        [
            {"id": "a", "Retry-After": "60"},
            {"id": "b", "metadata": {"retryAfter": "90"}},
            {"id": "c", "metadata": {"response_headers": {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}}},
        ],
        sample_limit=2,
    )

    assert summary["sources_with_retry_after_hint"] == 3
    assert summary["value_type_counts"] == {"http-date": 1, "seconds": 2}
    assert summary["samples"] == [
        {"source_id": "a", "value": "60", "value_type": "seconds"},
        {"source_id": "b", "value": "90", "value_type": "seconds"},
    ]
