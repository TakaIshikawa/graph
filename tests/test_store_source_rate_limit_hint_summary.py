from graph.store import summarize_source_rate_limit_hints


def test_source_rate_limit_hint_summary_normalizes_headers_and_samples_low_remaining():
    summary = summarize_source_rate_limit_hints(
        [
            {"source_id": "a", "provider": "api", "headers": {"X-RateLimit-Remaining": "2", "X-RateLimit-Limit": "10"}},
            {"source_id": "b", "metadata": {"retry_after": "60", "provider": "api"}},
            {"source_id": "c", "metadata": {"response_headers": {"x-ratelimit-remaining": "30"}}},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_rate_limit_hints"] == 3
    assert summary["sources_without_rate_limit_hints"] == 1
    assert summary["key_counts"] == {"retry_after": 1, "x-ratelimit-limit": 1, "x-ratelimit-remaining": 2}
    assert summary["provider_counts"] == {"api": 2}
    assert summary["samples"] == [{"source_id": "a", "provider": "api", "hints": {"x-ratelimit-limit": "10", "x-ratelimit-remaining": "2"}}]
