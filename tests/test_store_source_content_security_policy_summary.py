from graph.store import summarize_source_content_security_policies


def test_source_content_security_policy_extracts_case_insensitive_headers():
    summary = summarize_source_content_security_policies(
        [
            {"id": "a", "headers": {"Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; img-src https:"}},
            {"id": "b", "metadata": {"response_headers": {"content-security-policy": "connect-src api.example.com; frame-ancestors 'none'; upgrade-insecure-requests"}}},
            {"id": "c", "url": "https://example.com"},
        ]
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_content_security_policy"] == 2
    assert summary["directive_counts"] == {
        "connect-src": 1,
        "default-src": 1,
        "frame-ancestors": 1,
        "img-src": 1,
        "script-src": 1,
        "upgrade-insecure-requests": 1,
    }
    assert summary["unsafe_directive_counts"] == {"script-src": 1}
    assert summary["unsafe_token_counts"] == {"unsafe-inline": 1}
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b"]


def test_source_content_security_policy_reads_top_level_and_omits_missing_samples():
    summary = summarize_source_content_security_policies(
        [
            {"id": "missing"},
            {"source_id": "inline", "content_security_policy": "script-src 'unsafe-eval'"},
            {"source_id": "report", "csp": "default-src 'self'"},
        ],
        sample_limit=1,
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_content_security_policy"] == 2
    assert summary["unsafe_directive_counts"] == {"script-src": 1}
    assert summary["unsafe_token_counts"] == {"unsafe-eval": 1}
    assert summary["samples"] == [
        {
            "source_id": "inline",
            "directive_count": 1,
            "directives": ["script-src"],
            "has_unsafe_inline": False,
            "has_unsafe_eval": True,
        }
    ]
