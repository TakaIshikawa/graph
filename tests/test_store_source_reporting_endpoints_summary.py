from graph.store import summarize_source_reporting_endpoints


def test_reporting_endpoints_summary_counts_endpoints_and_malformed_entries():
    summary = summarize_source_reporting_endpoints(
        [
            {"source_id": "b", "Reporting-Endpoints": 'default="https://reports.example.test/a", csp="http://reports.example.test/csp"'},
            {"source_id": "a", "metadata": {"response_headers": {"reporting-endpoints": 'default="https://cdn.example.test/r,with-comma"'}}},
            {"source_id": "c", "headers": {"Reporting_Endpoints": "broken, nel=no-scheme"}},
            {"source_id": "d"},
        ],
        sample_limit=2,
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_reporting_endpoints"] == 3
    assert summary["endpoint_name_counts"] == {"csp": 1, "default": 2}
    assert summary["https_endpoint_count"] == 2
    assert summary["non_https_endpoint_count"] == 1
    assert summary["malformed_count"] == 2
    assert summary["missing_count"] == 1
    assert [row["source_id"] for row in summary["samples"]] == ["a", "b"]


def test_reporting_endpoints_summary_allows_empty_and_negative_sample_limit():
    summary = summarize_source_reporting_endpoints([{"source_id": "a"}], sample_limit=-1)

    assert summary["missing_count"] == 1
    assert summary["samples"] == []
