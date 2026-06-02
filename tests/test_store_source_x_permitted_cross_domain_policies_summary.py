from graph.store import summarize_source_x_permitted_cross_domain_policies


def test_x_permitted_cross_domain_policies_summary_counts_known_and_unknown():
    summary = summarize_source_x_permitted_cross_domain_policies(
        [
            {"source_id": "a", "X-Permitted-Cross-Domain-Policies": "none"},
            {"source_id": "b", "metadata": {"headers": {"x-permitted-cross-domain-policies": "ALL"}}},
            {"source_id": "c", "response_headers": {"x_permitted_cross_domain_policies": "partner-only"}},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_header"] == 3
    assert summary["policy_counts"] == {"all": 1, "none": 1, "partner-only": 1}
    assert summary["missing_header_count"] == 1
    assert summary["permissive_count"] == 1
    assert summary["unknown_value_count"] == 1
    assert [sample["source_id"] for sample in summary["samples"]] == ["b", "c"]
