from graph.store import summarize_source_access_control_allow_origins


def test_allow_origin_summary_counts_wildcard_null_echo_credentials_and_missing():
    summary = summarize_source_access_control_allow_origins(
        [
            {"source_id": "a", "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": "true"},
            {"source_id": "b", "metadata": {"access_control_allow_origin": "https://example.test"}},
            {"source_id": "c", "headers": {"Access-Control-Allow-Origin": "null"}},
            {
                "source_id": "d",
                "headers": {"Origin": "https://app.test"},
                "response_headers": {"access-control-allow-origin": "https://app.test"},
            },
            {"source_id": "e"},
        ]
    )

    assert summary["value_counts"]["*"] == 1
    assert summary["wildcard_count"] == 1
    assert summary["null_origin_count"] == 1
    assert summary["credential_wildcard_conflict_count"] == 1
    assert summary["origin_echo_count"] == 1
    assert summary["missing_origin_count"] == 1
    assert summary["domain_counts"] == {"app.test": 1, "example.test": 1}
