from graph.store import summarize_source_access_control_allow_credentials


def test_allow_credentials_summary_counts_true_invalid_and_missing():
    summary = summarize_source_access_control_allow_credentials(
        [
            {"source_id": "a", "Access-Control-Allow-Credentials": " TRUE "},
            {"source_id": "b", "response_headers": {"access_control_allow_credentials": "false"}},
            {"source_id": "c", "metadata": {"headers": {"Access-Control-Allow-Credentials": "yes"}}},
            {"source_id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["true_count"] == 1
    assert summary["false_or_invalid_count"] == 2
    assert summary["missing_access_control_allow_credentials_count"] == 1
    assert [row["source_id"] for row in summary["rows"]] == ["a", "b", "c"]
