from graph.store import summarize_source_cross_origin_resource_policies


def test_cross_origin_resource_policy_summary_counts_known_invalid_and_missing_values():
    summary = summarize_source_cross_origin_resource_policies(
        [
            {"source_id": "a", "Cross-Origin-Resource-Policy": "same-origin"},
            {"source_id": "b", "metadata": {"headers": {"cross-origin-resource-policy": "same-site"}}},
            {"source_id": "c", "response_headers": {"Cross-Origin-Resource-Policy": "cross-origin"}},
            {"source_id": "d", "metadata": {"Cross-Origin-Resource-Policy": "Same-Site"}},
            {"source_id": "e", "headers": {"cross_origin_resource_policy": "allow-all"}},
            {"source_id": "f", "metadata": {"cross_origin_resource_policy": "bad"}},
            {"source_id": "g"},
        ],
        sample_limit=5,
    )

    assert summary["sources_with_policy"] == 6
    assert summary["policy_counts"] == {"cross-origin": 1, "same-origin": 1, "same-site": 2}
    assert summary["bucket_counts"] == {"same-origin": 1, "same-site": 2, "cross-origin": 1, "invalid": 2, "missing": 1}
    assert summary["missing_policy_count"] == 1
    assert summary["permissive_count"] == 1
    assert summary["invalid_value_count"] == 2
    assert summary["invalid_values"] == [
        {"value": "allow-all", "count": 1, "source_ids": ["e"]},
        {"value": "bad", "count": 1, "source_ids": ["f"]},
    ]
    assert summary["samples"] == [
        {"source_id": "c", "policy": "cross-origin"},
        {"source_id": "e", "policy": "allow-all"},
        {"source_id": "f", "policy": "bad"},
    ]
