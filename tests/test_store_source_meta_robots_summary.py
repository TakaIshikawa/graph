from graph.store import summarize_source_meta_robots


def test_meta_robots_summary_counts_normalized_directives_and_html_meta():
    summary = summarize_source_meta_robots(
        [
            {"source_id": "b", "metadata": {"meta_robots": "NoIndex, nofollow"}},
            {"source_id": "a", "content": '<html><meta name="robots" content="noarchive, max-snippet:0"></html>'},
            {"source_id": "c", "robots_meta": "index, follow"},
            {"source_id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_meta_robots"] == 3
    assert summary["directive_counts"] == {"follow": 1, "index": 1, "max-snippet:0": 1, "noarchive": 1, "nofollow": 1, "noindex": 1}
    assert summary["noindex_count"] == 1
    assert summary["nofollow_count"] == 1
    assert summary["noarchive_count"] == 1
    assert summary["missing_meta_robots_count"] == 1
    assert summary["samples"][0] == {"source_id": "a", "directives": ["noarchive", "max-snippet:0"], "value": "noarchive, max-snippet:0"}
