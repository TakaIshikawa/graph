from graph.store import summarize_source_digest_headers


def test_digest_summary_counts_algorithms_strength_and_malformed_values():
    summary = summarize_source_digest_headers(
        [
            {"source_id": "a", "Digest": "sha-256=abc, sha-512=def"},
            {"source_id": "b", "metadata": {"headers": {"DIGEST": "md5=old"}}},
            {"source_id": "c", "response_headers": {"digest": "broken"}},
            {"source_id": "d"},
        ],
        sample_limit=1,
    )

    assert summary["sources_with_digest"] == 3
    assert summary["sources_missing_digest"] == 1
    assert summary["algorithm_counts"] == {"md5": 1, "sha-256": 1, "sha-512": 1}
    assert summary["strong_digest_count"] == 2
    assert summary["weak_or_legacy_digest_count"] == 1
    assert summary["invalid_digest_samples"] == [{"source_id": "c", "value": "broken"}]
