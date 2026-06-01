from graph.store import summarize_source_cache_headers


def test_source_cache_header_summary_reads_fields_and_counts_directives():
    summary = summarize_source_cache_headers(
        [
            {"source_id": "a", "Cache-Control": "max-age=60, PUBLIC", "etag": '"a"'},
            {"source_id": "b", "metadata": {"headers": {"cache-control": "no-cache", "Age": "7"}}},
            {"source_id": "c", "headers": {"Expires": "Wed, 21 Oct 2015 07:28:00 GMT"}},
            {"source_id": "d"},
        ],
        sample_limit=2,
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_cache_headers"] == 3
    assert summary["missing_header_count"] == 1
    assert summary["directive_counts"] == {"max-age": 1, "no-cache": 1, "public": 1}
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b"]


def test_source_cache_header_summary_finds_metadata_and_response_headers():
    summary = summarize_source_cache_headers([{"id": "s", "metadata": {"last_modified": "yesterday", "response_headers": {"ETag": '"x"'}}}])

    assert summary["samples"] == [{"source_id": "s", "headers": {"etag": '"x"', "last-modified": "yesterday"}}]
