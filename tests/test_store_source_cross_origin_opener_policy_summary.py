from graph.store import summarize_source_cross_origin_opener_policies


def test_cross_origin_opener_policy_summary_counts_unsafe_none():
    summary = summarize_source_cross_origin_opener_policies(
        [
            {"source_id": "a", "Cross-Origin-Opener-Policy": "same-origin"},
            {"source_id": "b", "metadata": {"headers": {"cross_origin_opener_policy": "Unsafe-None"}}},
            {"source_id": "c", "response_headers": {"Cross-Origin-Opener-Policy": "same-origin-allow-popups"}},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_policy"] == 3
    assert summary["policy_counts"] == {"same-origin": 1, "same-origin-allow-popups": 1, "unsafe-none": 1}
    assert summary["missing_policy_count"] == 1
    assert summary["unsafe_none_count"] == 1
    assert summary["samples"] == [{"source_id": "b", "policy": "unsafe-none", "field": "metadata.headers.cross_origin_opener_policy"}]
