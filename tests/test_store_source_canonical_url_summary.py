from graph.store import summarize_source_canonical_urls


def test_canonical_url_summary_finds_fields_metadata_and_html_links():
    summary = summarize_source_canonical_urls(
        [
            {"source_id": "b", "url": "https://example.test/b", "canonical_url": "https://example.test/a"},
            {"source_id": "a", "url": "https://example.test/a", "content": '<link rel="canonical" href="https://Example.test/a">'},
            {"source_id": "c", "url": "https://origin.test/c", "metadata": {"rel_canonical": "https://external.test/c"}},
            {"source_id": "d", "canonical": "https://example.test/a"},
            {"source_id": "e"},
        ]
    )

    assert summary["sources_with_canonical_url"] == 4
    assert summary["matching_source_url_count"] == 1
    assert summary["external_canonical_count"] == 1
    assert summary["missing_canonical_url_count"] == 1
    assert summary["canonical_domain_counts"] == {"example.test": 3, "external.test": 1}
    assert summary["duplicate_canonical_url_count"] == 1
    assert summary["samples"][0]["source_id"] == "a"
