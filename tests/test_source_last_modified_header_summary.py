from graph.store.source_last_modified_header_summary import summarize_source_last_modified_headers


def test_last_modified_summary_aggregates_rfc_1123_dates():
    summary = summarize_source_last_modified_headers(
        [
            {"source_id": "a", "Last-Modified": "Tue, 01 Jan 2030 05:30:00 GMT"},
            {"source_id": "b", "Last-Modified": "Tue, 01 Jan 2030 23:59:59 GMT"},
            {"source_id": "c", "Last-Modified": "Wed, 02 Jan 2030 00:00:00 GMT"},
        ]
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_last_modified"] == 3
    assert summary["missing_last_modified_count"] == 0
    assert summary["invalid_last_modified_count"] == 0
    assert summary["date_counts"] == {"2030-01-01": 2, "2030-01-02": 1}
    assert summary["rows"] == [
        {
            "date": "2030-01-01",
            "count": 2,
            "source_ids": ["a", "b"],
            "examples": ["Tue, 01 Jan 2030 05:30:00 GMT", "Tue, 01 Jan 2030 23:59:59 GMT"],
        },
        {
            "date": "2030-01-02",
            "count": 1,
            "source_ids": ["c"],
            "examples": ["Wed, 02 Jan 2030 00:00:00 GMT"],
        },
    ]


def test_last_modified_summary_counts_invalid_values_without_raising():
    summary = summarize_source_last_modified_headers(
        [
            {"source_id": "bad", "Last-Modified": "not a date"},
            {"source_id": "good", "Last-Modified": "Thu, 03 Jan 2030 12:00:00 GMT"},
        ]
    )

    assert summary["sources_with_last_modified"] == 2
    assert summary["invalid_last_modified_count"] == 1
    assert summary["invalid_examples"] == [{"source_id": "bad", "value": "not a date"}]
    assert summary["rows"] == [
        {
            "date": "2030-01-03",
            "count": 1,
            "source_ids": ["good"],
            "examples": ["Thu, 03 Jan 2030 12:00:00 GMT"],
        }
    ]


def test_last_modified_summary_reads_metadata_and_nested_headers():
    summary = summarize_source_last_modified_headers(
        [
            {"source_id": "direct", "last_modified": "Fri, 04 Jan 2030 00:00:00 GMT"},
            {"source_id": "nested", "response_headers": {"LAST-MODIFIED": "Sat, 05 Jan 2030 00:00:00 GMT"}},
            {"source_id": "metadata", "metadata": {"headers": {"last_modified": "Sun, 06 Jan 2030 00:00:00 GMT"}}},
        ]
    )

    assert summary["sources_with_last_modified"] == 3
    assert summary["date_counts"] == {"2030-01-04": 1, "2030-01-05": 1, "2030-01-06": 1}


def test_last_modified_summary_counts_missing_values():
    summary = summarize_source_last_modified_headers(
        [
            {"source_id": "missing"},
            {"source_id": "blank", "Last-Modified": ""},
            {"source_id": "present", "Last-Modified": "Mon, 07 Jan 2030 00:00:00 GMT"},
        ]
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_last_modified"] == 1
    assert summary["missing_last_modified_count"] == 2


def test_last_modified_summary_bounds_samples_with_sample_limit():
    summary = summarize_source_last_modified_headers(
        [
            {"source_id": "a", "Last-Modified": "Tue, 08 Jan 2030 00:00:00 GMT"},
            {"source_id": "b", "Last-Modified": "Tue, 08 Jan 2030 01:00:00 GMT"},
            {"source_id": "c", "Last-Modified": "Tue, 08 Jan 2030 02:00:00 GMT"},
            {"source_id": "bad1", "Last-Modified": "bad one"},
            {"source_id": "bad2", "Last-Modified": "bad two"},
        ],
        sample_limit=2,
    )

    assert summary["rows"] == [
        {
            "date": "2030-01-08",
            "count": 3,
            "source_ids": ["a", "b"],
            "examples": ["Tue, 08 Jan 2030 00:00:00 GMT", "Tue, 08 Jan 2030 01:00:00 GMT"],
        }
    ]
    assert summary["invalid_examples"] == [
        {"source_id": "bad1", "value": "bad one"},
        {"source_id": "bad2", "value": "bad two"},
    ]
