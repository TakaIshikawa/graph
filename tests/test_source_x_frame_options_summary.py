from graph.store.source_x_frame_options_summary import summarize_source_x_frame_options


def test_x_frame_options_summary_normalizes_known_policies_from_nested_headers():
    summary = summarize_source_x_frame_options(
        [
            {"source_id": "b", "metadata": {"headers": {"x_frame_options": "sameorigin"}}},
            {"source_id": "a", "X-Frame-Options": "DENY"},
            {"source_id": "c", "response_headers": {"X-Frame-Options": "allow-from https://example.test"}},
        ]
    )

    assert summary["policy_counts"] == {"allow-from": 1, "deny": 1, "sameorigin": 1}
    assert summary["source_ids"] == ["a", "b", "c"]


def test_x_frame_options_summary_ignores_missing_and_surfaces_unknown_values():
    summary = summarize_source_x_frame_options(
        [
            {"source_id": "unknown", "headers": {"X-Frame-Options": "maybe"}},
            {"source_id": "missing"},
        ]
    )

    assert summary["sources_with_header"] == 1
    assert summary["missing_header_count"] == 1
    assert summary["unknown_value_count"] == 1
    assert summary["unknown_values"] == [{"value": "maybe", "count": 1, "source_ids": ["unknown"]}]
