from graph.store import summarize_source_privacy_terms_links


def test_privacy_terms_policy_summary_counts_policy_cues_and_urls():
    summary = summarize_source_privacy_terms_links(
        [
            {"source_id": "a", "content": "Privacy Policy https://example.com/privacy and Terms of Service https://example.com/terms"},
            {"source_id": "b", "metadata": {"links": ["https://example.com/dpa", "Subprocessors and cookie policy"]}},
            {"source_id": "c", "content": "We use acceptable words in ordinary text."},
        ]
    )

    assert summary["sources_with_policy_hints"] == 2
    assert summary["cue_counts"] == {"cookie": 1, "dpa": 1, "privacy": 1, "subprocessor": 1, "terms": 1}
    assert summary["policy_url_samples"] == ["https://example.com/privacy", "https://example.com/terms", "https://example.com/dpa"]
