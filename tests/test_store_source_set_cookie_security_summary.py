from graph.store import summarize_source_set_cookie_security


def test_set_cookie_security_summary_counts_list_delimited_flags_and_safe_samples():
    summary = summarize_source_set_cookie_security(
        [
            {"source_id": "a", "Set-Cookie": ["sid=secret; Secure; HttpOnly; SameSite=Lax", "prefs=dark; Partitioned"]},
            {"source_id": "b", "metadata": {"set_cookie": "token=abc; Secure; SameSite=None\nplain=value"}},
            {"source_id": "c", "headers": {"Set-Cookie": "legacy=x; HttpOnly, temp=y; SameSite=Strict"}},
            {"source_id": "d"},
        ]
    )

    assert summary["cookie_count"] == 6
    assert summary["secure_count"] == 2
    assert summary["httponly_count"] == 2
    assert summary["samesite_counts"] == {"lax": 1, "none": 1, "strict": 1}
    assert summary["missing_secure_count"] == 4
    assert summary["missing_httponly_count"] == 4
    assert summary["missing_samesite_count"] == 3
    assert summary["partitioned_count"] == 1
    assert summary["missing_set_cookie_count"] == 1
    assert summary["samples"][0]["cookie_name"] == "sid"
    assert "secret" not in str(summary["samples"])
