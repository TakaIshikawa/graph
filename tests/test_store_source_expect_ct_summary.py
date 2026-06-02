from graph.store import summarize_source_expect_ct_headers


def test_expect_ct_summary_parses_directives_and_counts_invalid_values():
    summary = summarize_source_expect_ct_headers(
        [
            {"source_id": "b", "Expect-CT": 'max-age=86400; enforce; report-uri="https://report.test/ct"'},
            {"source_id": "a", "metadata": {"response_headers": {"expect-ct": "max-age=0"}}},
            {"source_id": "c", "headers": {"Expect-CT": "max-age=abc"}},
            {"source_id": "d", "expect_ct": "enforce"},
            {"source_id": "e"},
        ]
    )

    assert summary["sources_with_expect_ct"] == 4
    assert summary["enforce_count"] == 2
    assert summary["report_uri_count"] == 1
    assert summary["missing_max_age_count"] == 1
    assert summary["invalid_max_age_count"] == 1
    assert summary["missing_header_count"] == 1
    assert summary["max_age_buckets"] == {"lt_30_days": 1, "zero": 1}
    assert summary["samples"][0]["source_id"] == "a"
