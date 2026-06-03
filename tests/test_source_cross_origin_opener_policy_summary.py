from graph.store.source_cross_origin_opener_policy_summary import summarize_source_cross_origin_opener_policies


def test_cross_origin_opener_policy_summary_counts_known_values_and_unknowns():
    summary = summarize_source_cross_origin_opener_policies(
        [
            {"source_id": "b", "metadata": {"headers": {"cross_origin_opener_policy": "Unsafe-None"}}},
            {"source_id": "a", "Cross-Origin-Opener-Policy": "same-origin"},
            {"source_id": "c", "response_headers": {"Cross-Origin-Opener-Policy": "mystery"}},
            {"source_id": "d"},
        ]
    )

    assert summary["policy_counts"] == {"same-origin": 1, "unsafe-none": 1}
    assert summary["unknown_value_count"] == 1
    assert summary["unknown_values"] == [{"value": "mystery", "count": 1, "source_ids": ["c"]}]
    assert summary["source_ids"] == ["a", "b", "c"]
