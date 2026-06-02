from graph.store import summarize_source_clear_site_data_headers


def test_clear_site_data_summary_parses_quoted_directives_and_malformed_values():
    summary = summarize_source_clear_site_data_headers(
        [
            {"source_id": "a", "Clear-Site-Data": '"cache", "cookies"'},
            {"source_id": "b", "metadata": {"headers": {"clear-site-data": '"*"'}}},
            {"source_id": "c", "response_headers": {"Clear-Site-Data": "cache, cookies"}},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_clear_site_data"] == 3
    assert summary["directive_counts"] == {"*": 1, "cache": 1, "cookies": 1}
    assert summary["wildcard_count"] == 1
    assert summary["malformed_count"] == 1
    assert summary["missing_header_count"] == 1
    assert [sample["source_id"] for sample in summary["samples"]] == ["b", "c"]
