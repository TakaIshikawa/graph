from graph.store import summarize_source_x_robots_tags


def test_x_robots_tag_summary_splits_colon_qualified_directives():
    summary = summarize_source_x_robots_tags(
        [
            {"id": "a", "X-Robots-Tag": "noindex, nofollow"},
            {"id": "b", "metadata": {"response_headers": {"x_robots_tag": "googlebot: noarchive, max-snippet=50"}}},
            {"id": "c"},
        ]
    )

    assert summary["sources_with_x_robots_tag"] == 2
    assert summary["missing_x_robots_tag_count"] == 1
    assert summary["directive_counts"] == {"max-snippet": 1, "noarchive": 1, "nofollow": 1, "noindex": 1}
