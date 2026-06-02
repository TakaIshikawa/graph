from graph.store import summarize_source_service_worker_allowed


def test_service_worker_allowed_summary_normalizes_scopes_and_export():
    summary = summarize_source_service_worker_allowed(
        [
            {"id": "a", "headers": {"Service-Worker-Allowed": "/"}},
            {"id": "b", "metadata": {"response_headers": {"service_worker_allowed": "app/"}}},
            {"id": "c"},
        ]
    )

    assert summary["sources_with_service_worker_allowed"] == 2
    assert summary["missing_service_worker_allowed_count"] == 1
    assert summary["scope_counts"] == {"/": 1, "/app/": 1}
    assert summary["broad_scope_count"] == 1
