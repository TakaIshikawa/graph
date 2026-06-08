from graph.store import summarize_source_last_modified


def test_source_last_modified_summary_counts_parseable_invalid_and_missing_values():
    summary = summarize_source_last_modified(
        [
            {"source_id": "a", "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT"},
            {"source_id": "b", "metadata": {"last_modified": "2020-01-01T00:00:00Z"}},
            {"source_id": "c", "metadata": {"last_modified": "not a date"}},
            {"source_id": "d"},
        ],
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_last_modified"] == 3
    assert summary["missing_last_modified_count"] == 1
    assert summary["parseable_last_modified_count"] == 2
    assert summary["unparseable_last_modified_count"] == 1
    assert summary["oldest_last_modified"] == "2015-10-21T07:28:00+00:00"
    assert summary["newest_last_modified"] == "2020-01-01T00:00:00+00:00"
    assert summary["rows"] == [
        {
            "source_id": "a",
            "last_modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            "parseable": True,
            "normalized_last_modified": "2015-10-21T07:28:00+00:00",
            "age_order_key": "2015-10-21T07:28:00+00:00",
        },
        {
            "source_id": "b",
            "last_modified": "2020-01-01T00:00:00Z",
            "parseable": True,
            "normalized_last_modified": "2020-01-01T00:00:00+00:00",
            "age_order_key": "2020-01-01T00:00:00+00:00",
        },
        {
            "source_id": "c",
            "last_modified": "not a date",
            "parseable": False,
            "normalized_last_modified": "",
            "age_order_key": "",
        },
        {
            "source_id": "d",
            "last_modified": "",
            "parseable": False,
            "normalized_last_modified": "",
            "age_order_key": "",
        },
    ]


def test_source_last_modified_summary_reads_nested_headers():
    summary = summarize_source_last_modified(
        [{"source_id": "nested", "metadata": {"response_headers": {"last-modified": "Tue, 15 Nov 1994 08:12:31 GMT"}}}]
    )

    assert summary["sources_with_last_modified"] == 1
    assert summary["rows"] == [
        {
            "source_id": "nested",
            "last_modified": "Tue, 15 Nov 1994 08:12:31 GMT",
            "parseable": True,
            "normalized_last_modified": "1994-11-15T08:12:31+00:00",
            "age_order_key": "1994-11-15T08:12:31+00:00",
        }
    ]


def test_source_last_modified_summary_limits_samples():
    summary = summarize_source_last_modified(
        [
            {"source_id": "a", "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT"},
            {"source_id": "b", "Last-Modified": "Thu, 22 Oct 2015 07:28:00 GMT"},
            {"source_id": "c", "Last-Modified": "Fri, 23 Oct 2015 07:28:00 GMT"},
        ],
        sample_limit=2,
    )

    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b"]
