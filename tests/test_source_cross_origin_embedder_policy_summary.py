from graph.store.source_cross_origin_embedder_policy_summary import summarize_source_cross_origin_embedder_policies


def test_cross_origin_embedder_policy_summary_counts_known_values_and_unknowns():
    summary = summarize_source_cross_origin_embedder_policies(
        [
            {"source_id": "b", "metadata": {"response_headers": {"Cross-Origin-Embedder-Policy": "credentialless"}}},
            {"source_id": "a", "cross_origin_embedder_policy": "require-corp"},
            {"source_id": "c", "headers": {"cross-origin-embedder-policy": "unexpected"}},
            {"source_id": "d"},
        ]
    )

    assert summary["policy_counts"] == {"credentialless": 1, "require-corp": 1}
    assert summary["isolating_policy_count"] == 2
    assert summary["unknown_value_count"] == 1
    assert summary["unknown_values"] == [{"value": "unexpected", "count": 1, "source_ids": ["c"]}]
    assert summary["source_ids"] == ["a", "b", "c"]
