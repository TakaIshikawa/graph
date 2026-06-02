from graph.store import summarize_source_resource_hints


def test_resource_hint_summary_counts_supported_relations_and_export():
    summary = summarize_source_resource_hints(
        [
            {"id": "a", "headers": {"Link": '<//dns.test>; rel=dns-prefetch, <https://api.test>; rel="preconnect"; crossorigin'}},
            {"id": "b", "metadata": {"response_headers": {"LINK": '<mod.js>; rel=modulepreload'}}},
            {"id": "c"},
        ]
    )

    assert summary["relation_counts"] == {"dns-prefetch": 1, "modulepreload": 1, "preconnect": 1}
    assert summary["cross_origin_count"] == 1
    assert [sample["relation"] for sample in summary["samples"]] == ["dns-prefetch", "preconnect", "modulepreload"]
