from graph.store import summarize_source_cookie_prefixes


def test_cookie_prefix_summary_validates_prefixes_and_export():
    summary = summarize_source_cookie_prefixes(
        [
            {"id": "a", "headers": {"Set-Cookie": "__Host-sid=1; Secure; Path=/"}},
            {"id": "b", "response_headers": {"set_cookie": "__Host-bad=1; Secure; Domain=example.com; Path=/"}},
            {"id": "c", "metadata": {"response_headers": {"Set-Cookie": "__Secure-token=1; Secure"}}},
            {"id": "d", "headers": {"Set-Cookie": ["__Secure-bad=1; Path=/", "plain=1"]}},
        ]
    )

    assert summary["host_prefix_count"] == 1
    assert summary["secure_prefix_count"] == 1
    assert summary["invalid_host_prefix_count"] == 1
    assert summary["invalid_secure_prefix_count"] == 1
    assert summary["unprefixed_cookie_count"] == 1
