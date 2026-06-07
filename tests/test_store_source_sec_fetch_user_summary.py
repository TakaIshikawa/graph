from graph.store import summarize_source_sec_fetch_users


def test_sec_fetch_user_summary_counts_lookup_variants_and_samples():
    summary = summarize_source_sec_fetch_users(
        [
            {"source_id": "a", "Sec-Fetch-User": "?1"},
            {"source_id": "b", "metadata": {"sec_fetch_user": "?0"}},
            {"source_id": "c", "headers": {"sec-fetch-user": "?1"}},
            {"source_id": "d", "response_headers": {"Sec-Fetch-User": "unexpected"}},
            {"source_id": "e"},
        ],
        sample_limit=3,
    )

    assert summary["total_sources"] == 5
    assert summary["sources_with_sec_fetch_user"] == 4
    assert summary["missing_sec_fetch_user_count"] == 1
    assert summary["value_counts"] == {"?0": 1, "?1": 2, "unexpected": 1}
    assert summary["user_activation_count"] == 2
    assert summary["non_user_activation_count"] == 1
    assert summary["unexpected_value_count"] == 1
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b", "c"]
