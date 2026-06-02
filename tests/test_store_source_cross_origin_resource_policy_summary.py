from graph.store import summarize_source_cross_origin_resource_policies


def test_cross_origin_resource_policy_summary_samples_permissive_values():
    summary = summarize_source_cross_origin_resource_policies(
        [
            {"source_id": "a", "Cross-Origin-Resource-Policy": "same-origin"},
            {"source_id": "b", "metadata": {"headers": {"cross-origin-resource-policy": "same-site"}}},
            {"source_id": "c", "response_headers": {"Cross-Origin-Resource-Policy": "cross-origin"}},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_policy"] == 3
    assert summary["policy_counts"] == {"cross-origin": 1, "same-origin": 1, "same-site": 1}
    assert summary["missing_policy_count"] == 1
    assert summary["permissive_count"] == 1
    assert summary["samples"] == [{"source_id": "c", "policy": "cross-origin"}]
