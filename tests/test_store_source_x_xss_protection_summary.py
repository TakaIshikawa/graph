from graph.store import summarize_source_x_xss_protections


def test_x_xss_protection_summary_parses_values_and_export():
    summary = summarize_source_x_xss_protections(
        [
            {"id": "a", "headers": {"X-XSS-Protection": "0"}},
            {"id": "b", "metadata": {"response_headers": {"x_xss_protection": "1; mode=block"}}},
            {"id": "c", "response_headers": {"X-XSS-PROTECTION": "1; report=https://example.test/report"}},
            {"id": "d", "headers": {"x-xss-protection": "maybe"}},
            {"id": "e"},
        ]
    )

    assert summary["enabled_count"] == 2
    assert summary["disabled_count"] == 1
    assert summary["block_mode_count"] == 1
    assert summary["report_uri_count"] == 1
    assert summary["invalid_value_count"] == 1
    assert summary["missing_x_xss_protection_count"] == 1
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b", "c", "d"]
