from graph.store.source_early_data_summary import summarize_source_early_data_headers


def test_early_data_summary_counts_replay_risk_unexpected_missing_and_samples():
    summary = summarize_source_early_data_headers(
        [
            {"source_id": "a", "Early-Data": 1},
            {"source_id": "b", "headers": {"early_data": " 0 "}},
            {"source_id": "c", "metadata": {"response_headers": {"EARLY-DATA": "maybe"}}},
            {"source_id": "d"},
        ],
        sample_limit=2,
    )

    assert summary["sources_with_early_data"] == 3
    assert summary["missing_early_data_count"] == 1
    assert summary["value_counts"] == {"0": 1, "1": 1, "maybe": 1}
    assert summary["replay_risk_count"] == 1
    assert summary["unexpected_value_count"] == 1
    assert summary["samples"] == [{"source_id": "a", "value": "1"}, {"source_id": "b", "value": "0"}]
