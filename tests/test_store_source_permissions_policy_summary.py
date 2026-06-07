from graph.store import summarize_source_permissions_policies


def test_permissions_policy_summary_counts_lookup_directives_risky_invalid_and_missing():
    summary = summarize_source_permissions_policies(
        [
            {"source_id": "a", "Permissions-Policy": "camera=(self), geolocation=()"},
            {"source_id": "b", "metadata": {"permissions_policy": "payment=(*)"}},
            {"source_id": "c", "headers": {"Permissions-Policy": "microphone self"}},
            {"source_id": "d", "response_headers": {"permissions-policy": "fullscreen=()"}},
            {"source_id": "e"},
        ]
    )

    assert summary["total_sources"] == 5
    assert summary["sources_with_policy"] == 4
    assert summary["missing_policy_count"] == 1
    assert summary["directive_counts"] == {"camera": 1, "fullscreen": 1, "geolocation": 1, "payment": 1}
    assert summary["risky_allowance_count"] == 2
    assert summary["invalid_fragment_count"] == 1
    assert summary["empty_policy_count"] == 1
