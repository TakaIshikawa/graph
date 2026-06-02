from graph.store import summarize_source_content_location_headers


def test_content_location_summary_classifies_urls_domains_and_invalid_values():
    summary = summarize_source_content_location_headers(
        [
            {"source_id": "a", "url": "https://example.com/page", "Content-Location": "https://example.com/canonical"},
            {"source_id": "b", "url": "https://example.com/page", "metadata": {"headers": {"CONTENT_LOCATION": "https://other.test/item"}}},
            {"source_id": "c", "response_headers": {"content-location": "/relative"}},
            {"source_id": "d", "Content-Location": "not a url"},
            {"source_id": "e"},
        ],
        sample_limit=1,
    )

    assert summary["sources_with_content_location"] == 4
    assert summary["sources_missing_content_location"] == 1
    assert summary["absolute_url_count"] == 2
    assert summary["relative_url_count"] == 1
    assert summary["same_domain_count"] == 1
    assert summary["cross_domain_count"] == 1
    assert summary["top_content_location_domains"] == {"example.com": 1, "other.test": 1}
    assert summary["invalid_content_location_samples"] == [{"source_id": "d", "value": "not a url"}]
