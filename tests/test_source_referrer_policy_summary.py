from graph.store.source_referrer_policy_summary import summarize_source_referrer_policies


def test_referrer_policy_summary_counts_tokens_and_effective_fallback():
    summary = summarize_source_referrer_policies(
        [
            {"source_id": "b", "metadata": {"headers": {"Referrer-Policy": "origin, strict-origin"}}},
            {"source_id": "a", "referrer_policy": "No-Referrer"},
            {"source_id": "c"},
        ]
    )

    assert summary["effective_policy_counts"] == {"no-referrer": 1, "strict-origin": 1}
    assert summary["token_counts"] == {"no-referrer": 1, "origin": 1, "strict-origin": 1}
    assert summary["missing_header_count"] == 1
    assert summary["source_ids"] == ["a", "b"]


def test_referrer_policy_summary_retains_invalid_tokens():
    summary = summarize_source_referrer_policies(
        [{"source_id": "bad", "response_headers": {"referrer-policy": "origin, nope"}}]
    )

    assert summary["invalid_token_count"] == 1
    assert summary["invalid_values"] == [{"value": "nope", "count": 1, "source_ids": ["bad"]}]
