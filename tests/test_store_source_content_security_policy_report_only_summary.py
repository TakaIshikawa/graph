from graph.store import summarize_source_content_security_policy_report_only


def test_csp_report_only_summary_counts_directives_reporting_and_unsafe_tokens():
    summary = summarize_source_content_security_policy_report_only(
        [
            {"source_id": "b", "Content-Security-Policy-Report-Only": "default-src 'self'; script-src 'unsafe-inline'; report-uri /csp"},
            {"source_id": "a", "metadata": {"response_headers": {"content-security-policy-report-only": "script-src 'unsafe-eval'; report-to endpoint"}}},
            {"source_id": "missing"},
        ],
        sample_limit=1,
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_content_security_policy_report_only"] == 2
    assert summary["missing_content_security_policy_report_only_count"] == 1
    assert summary["directive_counts"] == {"default-src": 1, "report-to": 1, "report-uri": 1, "script-src": 2}
    assert summary["report_uri_counts"] == {"/csp": 1}
    assert summary["report_to_counts"] == {"endpoint": 1}
    assert summary["unsafe_directive_counts"] == {"script-src": 2}
    assert summary["unsafe_token_counts"] == {"unsafe-eval": 1, "unsafe-inline": 1}
    assert summary["samples"][0]["source_id"] == "a"
