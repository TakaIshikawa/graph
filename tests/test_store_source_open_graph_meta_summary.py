from graph.store import summarize_source_open_graph_meta


def test_open_graph_summary_reads_metadata_and_html_meta_tags():
    summary = summarize_source_open_graph_meta(
        [
            {"id": "a", "metadata": {"open_graph": {"title": "A", "image": "a.png"}}},
            {"id": "b", "metadata": {"html": '<meta property="og:title" content="B"><meta name="og:image" content="b.png"><meta property="og:title" content="B2">'}},
            {"id": "c"},
        ]
    )

    assert summary["sources_with_open_graph"] == 2
    assert summary["property_counts"] == {"og:image": 2, "og:title": 2}
    assert summary["common_property_missing_counts"]["og:type"] == 3
    assert summary["samples"][0] == {"source_id": "a", "property": "og:title", "content": "A"}
