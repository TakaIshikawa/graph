from graph.store.source_content_location_summary import summarize_source_content_locations


def test_content_location_summary_reads_nested_headers_and_direct_metadata():
    summary = summarize_source_content_locations(
        [
            {"source_id": "direct", "Content-Location": "https://Example.COM/canonical"},
            {"source_id": "nested", "response_headers": {"content_location": "/variant"}},
            {"source_id": "metadata", "metadata": {"content_location": "../local-copy"}},
            {"source_id": "blank", "metadata": {"headers": {"Content-Location": "   "}}},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_content_location"] == 3
    assert summary["missing_content_location_count"] == 1
    assert summary["kind_counts"] == {"absolute_url": 1, "relative_path": 1, "root_relative_path": 1}
    assert summary["hostname_counts"] == {"example.com": 1}


def test_content_location_summary_classifies_relative_malformed_and_samples():
    summary = summarize_source_content_locations(
        [
            {"source_id": "abs", "headers": {"CONTENT-LOCATION": "https://cdn.EXAMPLE.test/a"}},
            {"source_id": "root", "Content-Location": "/assets/a.json"},
            {"source_id": "relative", "Content-Location": "./copy.json"},
            {"source_id": "plain-relative", "Content-Location": "assets/copy.json"},
            {"source_id": "other", "Content-Location": "not a url"},
        ],
        sample_limit=3,
    )

    assert summary["kind_counts"] == {
        "absolute_url": 1,
        "other": 1,
        "relative_path": 2,
        "root_relative_path": 1,
    }
    assert summary["hostname_counts"] == {"cdn.example.test": 1}
    assert summary["samples"] == [
        {"source_id": "abs", "kind": "absolute_url", "value": "https://cdn.EXAMPLE.test/a"},
        {"source_id": "root", "kind": "root_relative_path", "value": "/assets/a.json"},
        {"source_id": "relative", "kind": "relative_path", "value": "./copy.json"},
    ]
