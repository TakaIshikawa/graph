from graph.store.source_www_authenticate_summary import summarize_source_www_authenticate_challenges


def test_www_authenticate_summary_counts_schemes_case_insensitively():
    summary = summarize_source_www_authenticate_challenges(
        [
            {"id": "bearer", "WWW-Authenticate": 'Bearer realm="api", error="invalid_token"'},
            {"id": "basic", "metadata": {"www_authenticate": "basic realm=docs"}},
            {"id": "digest", "headers": {"www-authenticate": 'Digest realm="docs", qop="auth,auth-int"'}},
            {"id": "negotiate", "response_headers": {"WWW-Authenticate": "NEGOTIATE"}},
        ]
    )

    assert summary["scheme_counts"] == {"basic": 1, "bearer": 1, "digest": 1, "negotiate": 1}
    assert summary["basic_count"] == 1
    assert summary["bearer_count"] == 1
    assert summary["digest_count"] == 1
    assert summary["negotiate_count"] == 1


def test_www_authenticate_summary_counts_realm_presence_without_values():
    summary = summarize_source_www_authenticate_challenges(
        [
            {"id": "one", "WWW-Authenticate": 'Basic realm="private-docs"'},
            {"id": "two", "WWW-Authenticate": 'Bearer realm="token-area", error_description="do not expose"'},
            {"id": "three", "WWW-Authenticate": "Bearer error=invalid_token"},
        ]
    )

    assert summary["realm_presence_counts"] == {"basic": 1, "bearer": 1}
    assert "private-docs" not in str(summary)
    assert "token-area" not in str(summary)
    assert "do not expose" not in str(summary)


def test_www_authenticate_summary_handles_missing_empty_and_malformed_values():
    summary = summarize_source_www_authenticate_challenges(
        [
            {"id": "missing"},
            {"id": "empty", "WWW-Authenticate": ""},
            {"id": "malformed", "WWW-Authenticate": 'realm="missing-scheme"'},
            {"id": "mixed", "WWW-Authenticate": 'Bearer realm="api", broken==, Basic realm="docs"'},
        ]
    )

    assert summary["sources_with_www_authenticate"] == 1
    assert summary["missing_header_count"] == 1
    assert summary["empty_value_count"] == 1
    assert summary["malformed_challenge_count"] == 2
    assert summary["scheme_counts"] == {"basic": 1, "bearer": 1}


def test_www_authenticate_summary_respects_sample_limit_after_counting():
    summary = summarize_source_www_authenticate_challenges(
        [{"id": "b", "WWW-Authenticate": "Bearer"}, {"id": "a", "WWW-Authenticate": "Basic"}],
        sample_limit=0,
    )

    assert summary["scheme_counts"] == {"basic": 1, "bearer": 1}
    assert summary["samples"] == []
