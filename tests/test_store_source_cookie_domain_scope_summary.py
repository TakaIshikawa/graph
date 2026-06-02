from graph.store import summarize_source_cookie_domain_scopes


def test_cookie_domain_scope_summary_counts_domain_scope_and_export():
    summary = summarize_source_cookie_domain_scopes(
        [
            {"id": "a", "headers": {"Set-Cookie": "host=1; Secure; Path=/"}},
            {"id": "b", "metadata": {"response_headers": {"set-cookie": "domain=1; Domain=.example.com; Path=/"}}},
            {"id": "c", "response_headers": {"Set-Cookie": ["broad=1; Domain=.com", "uk=1; Domain=.co.uk"]}},
        ]
    )

    assert summary["host_only_cookie_count"] == 1
    assert summary["domain_cookie_count"] == 3
    assert summary["public_suffix_like_domain_count"] == 2
    assert [sample["classification"] for sample in summary["samples"]] == ["host_only", "domain", "public_suffix_like", "public_suffix_like"]
