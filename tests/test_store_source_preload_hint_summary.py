from graph.store import summarize_source_preload_hints


def test_preload_hint_summary_parses_link_headers_and_export():
    summary = summarize_source_preload_hints(
        [
            {"id": "a", "headers": {"Link": '<https://cdn.test/app.js>; rel="preload"; as=script; crossorigin'}},
            {"id": "b", "metadata": {"response_headers": {"link": '<style.css>; rel=preload; as=style, <font.woff2>; rel="preload"; crossorigin=anonymous'}}},
            {"id": "c", "headers": {"Link": "<x>; rel=preconnect"}},
        ]
    )

    assert summary["sources_with_preload"] == 2
    assert summary["as_counts"] == {"script": 1, "style": 1}
    assert summary["missing_as_count"] == 1
    assert summary["cross_origin_count"] == 2
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b", "b"]
