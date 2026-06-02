from graph.store import summarize_source_content_digests


def test_content_digest_summary_parses_digest_headers_and_invalid_values():
    summary = summarize_source_content_digests(
        [
            {"id": "a", "Content-Digest": "sha-256=abc, sha-512=def"},
            {"id": "b", "metadata": {"response_headers": {"Digest": "md5=ghi"}}},
            {"id": "c", "headers": {"CONTENT_DIGEST": "broken"}},
            {"id": "d"},
        ]
    )

    assert summary["sources_with_content_digest"] == 3
    assert summary["missing_content_digest_count"] == 1
    assert summary["invalid_digest_count"] == 1
    assert summary["algorithm_counts"] == {"md5": 1, "sha-256": 1, "sha-512": 1}
