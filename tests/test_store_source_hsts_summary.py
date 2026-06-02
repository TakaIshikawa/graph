from graph.store.source_hsts_summary import summarize_source_hsts_policies


def test_hsts_summary_buckets_max_age_and_counts_directives():
    summary = summarize_source_hsts_policies(
        [
            {"source_id": "a", "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload"},
            {"source_id": "b", "headers": {"strict_transport_security": "MAX-AGE=3600; max-age=1"}},
            {"source_id": "c", "metadata": {"response_headers": {"STRICT-TRANSPORT-SECURITY": "max-age=abc"}}},
            {"source_id": "d", "Strict-Transport-Security": "includeSubDomains"},
            {"source_id": "e"},
        ],
        sample_limit=2,
    )

    assert summary["sources_with_hsts"] == 4
    assert summary["max_age_buckets"] == {"lt_1_day": 1, "gte_1_year": 1}
    assert summary["include_subdomains_count"] == 2
    assert summary["preload_count"] == 1
    assert summary["missing_max_age_count"] == 1
    assert summary["invalid_max_age_count"] == 1
    assert summary["missing_hsts_count"] == 1
    assert summary["samples"] == [
        {"source_id": "a", "value": "max-age=31536000; includeSubDomains; preload", "max_age": "31536000"},
        {"source_id": "b", "value": "MAX-AGE=3600; max-age=1", "max_age": "3600"},
    ]
