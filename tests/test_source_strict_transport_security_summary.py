from graph.store.source_strict_transport_security_summary import summarize_source_strict_transport_security


def test_strict_transport_security_summary_parses_directives_case_insensitively():
    summary = summarize_source_strict_transport_security(
        [
            {"source_id": "b", "metadata": {"headers": {"Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload"}}},
            {"source_id": "a", "strict_transport_security": "max-age=0"},
            {"source_id": "c"},
        ]
    )

    assert summary["max_age_seconds"] == [0, 31536000]
    assert summary["include_subdomains_count"] == 1
    assert summary["preload_count"] == 1
    assert summary["missing_header_count"] == 1
    assert summary["source_ids"] == ["a", "b"]


def test_strict_transport_security_summary_counts_missing_and_invalid_max_age():
    summary = summarize_source_strict_transport_security(
        [
            {"source_id": "missing-age", "headers": {"strict-transport-security": "includeSubDomains"}},
            {"source_id": "bad-age", "response_headers": {"Strict-Transport-Security": "max-age=abc"}},
        ]
    )

    assert summary["missing_max_age_count"] == 1
    assert summary["invalid_max_age_count"] == 1
