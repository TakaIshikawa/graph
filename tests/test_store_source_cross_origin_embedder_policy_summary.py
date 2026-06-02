from graph.store import summarize_source_cross_origin_embedder_policies


def test_cross_origin_embedder_policy_summary_counts_isolating_and_weak():
    summary = summarize_source_cross_origin_embedder_policies(
        [
            {"source_id": "a", "cross_origin_embedder_policy": "require-corp"},
            {"source_id": "b", "metadata": {"response_headers": {"Cross-Origin-Embedder-Policy": "credentialless"}}},
            {"source_id": "c", "headers": {"cross-origin-embedder-policy": "unsafe-none"}},
            {"source_id": "d"},
        ],
        sample_limit=1,
    )

    assert summary["sources_with_policy"] == 3
    assert summary["policy_counts"] == {"credentialless": 1, "require-corp": 1, "unsafe-none": 1}
    assert summary["missing_policy_count"] == 1
    assert summary["isolating_policy_count"] == 2
    assert summary["weak_policy_count"] == 1
    assert summary["samples"] == [{"source_id": "c", "policy": "unsafe-none", "field": "headers.cross-origin-embedder-policy"}]
