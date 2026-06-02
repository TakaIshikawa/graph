from graph.store import summarize_source_favicon_hints


def test_favicon_hint_summary_detects_metadata_and_simple_html_links():
    summary = summarize_source_favicon_hints(
        [
            {"source_id": "b", "url": "https://example.test/page", "favicon_url": "https://example.test/favicon.ico"},
            {"source_id": "a", "url": "https://example.test/page", "content": '<link rel="apple-touch-icon" href="https://cdn.test/apple.PNG">'},
            {"source_id": "c", "metadata": {"icon_url": "/icon.svg"}},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_favicon_hint"] == 3
    assert summary["icon_relation_counts"] == {"apple-touch-icon": 1, "icon": 2}
    assert summary["external_icon_count"] == 1
    assert summary["missing_favicon_hint_count"] == 1
    assert summary["icon_extension_counts"] == {"ico": 1, "png": 1, "svg": 1}
    assert summary["samples"][0] == {"source_id": "a", "relation": "apple-touch-icon", "icon_url": "https://cdn.test/apple.PNG"}
