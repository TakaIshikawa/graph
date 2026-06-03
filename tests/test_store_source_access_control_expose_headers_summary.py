from graph.store import summarize_source_access_control_expose_headers


def test_expose_headers_summary_counts_headers_wildcard_and_samples():
    summary = summarize_source_access_control_expose_headers(
        [
            {"source_id": "b", "Access-Control-Expose-Headers": "ETag, X-Trace"},
            {"source_id": "a", "metadata": {"response_headers": {"access-control-expose-headers": "*"}}},
            {"source_id": "c", "headers": {"Access_Control_Expose_Headers": "etag"}},
            {"source_id": "d"},
        ],
        sample_limit=2,
    )

    assert summary["total_sources"] == 4
    assert summary["exposed_header_counts"] == {"etag": 2, "x-trace": 1}
    assert summary["wildcard_count"] == 1
    assert summary["missing_access_control_expose_headers_count"] == 1
    assert [row["source_id"] for row in summary["samples"]] == ["a", "b"]
