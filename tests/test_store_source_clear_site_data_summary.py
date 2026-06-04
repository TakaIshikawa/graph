from graph.store import summarize_source_clear_site_data_headers


def test_clear_site_data_summary_parses_quoted_directives_and_malformed_values():
    summary = summarize_source_clear_site_data_headers(
        [
            {"source_id": "a", "Clear-Site-Data": '"cache", "cookies", "storage", "executionContexts"'},
            {"source_id": "b", "metadata": {"headers": {"clear-site-data": '"*"'}}},
            {"source_id": "c", "response_headers": {"Clear-Site-Data": "cache, cookies"}},
            {"source_id": "d", "clear_site_data": ""},
            {"source_id": "e"},
        ],
        sample_limit=2,
    )

    assert summary["total_sources"] == 5
    assert summary["sources_with_clear_site_data"] == 4
    assert summary["directive_counts"] == {"*": 1, "cache": 1, "cookies": 1, "executioncontexts": 1, "storage": 1}
    assert summary["wildcard_count"] == 1
    assert summary["malformed_count"] == 2
    assert summary["missing_clear_site_data_count"] == 1
    assert [sample["source_id"] for sample in summary["samples"]] == ["b", "c"]


def test_clear_site_data_summary_sample_limit_can_suppress_samples():
    summary = summarize_source_clear_site_data_headers(
        [
            {"source_id": "a", "Clear-Site-Data": '"*"'},
            {"source_id": "b", "Clear-Site-Data": "cache"},
            {"source_id": "c", "Clear-Site-Data": ""},
            {"source_id": "d"},
        ],
        sample_limit=0,
    )

    assert summary["sources_with_clear_site_data"] == 3
    assert summary["wildcard_count"] == 1
    assert summary["malformed_count"] == 2
    assert summary["missing_clear_site_data_count"] == 1
    assert summary["samples"] == []
