from graph.store import summarize_source_authentication_hints


def test_source_authentication_hints_detects_and_redacts_common_types():
    summary = summarize_source_authentication_hints(
        [
            {"id": "a", "headers": {"Authorization": "Bearer abc", "Cookie": "sid=x"}},
            {"id": "b", "api_key": "secret"},
            {"id": "c", "url": "https://example.com"},
        ]
    )
    assert summary["total_sources"] == 3
    assert summary["sources_with_auth_hints"] == 2
    assert summary["hint_type_counts"] == {"api_key": 1, "bearer": 1, "cookie": 1}
    assert summary["severity_counts"]["high"] == 3
    assert all(sample["value"] == "[redacted]" for sample in summary["samples"])


def test_source_authentication_hints_detects_basic_and_token_like_metadata():
    summary = summarize_source_authentication_hints([{"source_id": "s", "metadata": {"auth_token": "Basic abc"}}])
    assert summary["hint_type_counts"] == {"basic": 1}
