from graph.store import summarize_source_www_authenticate_headers


def test_www_authenticate_summary_counts_schemes_and_nested_headers():
    summary = summarize_source_www_authenticate_headers(
        [
            {"id": "bearer", "WWW-Authenticate": 'Bearer realm="api"'},
            {"id": "basic", "metadata": {"www_authenticate": "Basic realm=docs"}},
            {"id": "digest", "headers": {"www-authenticate": "Digest realm=docs"}},
            {"id": "multi", "response_headers": {"WWW-Authenticate": ["Negotiate", "Custom token"]}},
            {"id": "nested", "metadata": {"response_headers": {"WWW_Authenticate": "Bearer error=invalid_token"}}},
            {"id": "empty", "WWW-Authenticate": ""},
            {"id": "missing"},
        ]
    )

    assert summary["total_sources"] == 7
    assert summary["sources_with_www_authenticate"] == 5
    assert summary["missing_header_count"] == 1
    assert summary["empty_value_count"] == 1
    assert summary["scheme_counts"] == {"basic": 1, "bearer": 2, "custom": 1, "digest": 1, "negotiate": 1}
    assert summary["bearer_count"] == 2
    assert summary["basic_count"] == 1
    assert summary["digest_count"] == 1
    assert summary["unknown_scheme_count"] == 1
    assert summary["samples"][0] == {"source_id": "basic", "schemes": ["basic"]}


def test_www_authenticate_summary_respects_sample_limit_while_counting():
    summary = summarize_source_www_authenticate_headers(
        [{"id": "b", "WWW-Authenticate": "Bearer"}, {"id": "a", "WWW-Authenticate": "Basic"}],
        sample_limit=0,
    )

    assert summary["bearer_count"] == 1
    assert summary["basic_count"] == 1
    assert summary["samples"] == []
