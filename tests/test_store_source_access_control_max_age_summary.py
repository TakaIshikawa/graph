from graph.store import summarize_source_access_control_max_ages


def test_max_age_summary_buckets_numeric_values_and_invalids():
    summary = summarize_source_access_control_max_ages(
        [
            {"source_id": "a", "Access-Control-Max-Age": "0"},
            {"source_id": "b", "response_headers": {"access-control-max-age": "600"}},
            {"source_id": "c", "metadata": {"headers": {"Access-Control-Max-Age": "7200"}}},
            {"source_id": "d", "Access-Control-Max-Age": "bad"},
            {"source_id": "e"},
        ]
    )

    assert summary["bucket_counts"] == {"0": 1, "1-600": 1, "3601-86400": 1}
    assert summary["invalid_value_count"] == 1
    assert summary["missing_access_control_max_age_count"] == 1
    assert summary["rows"][0]["source_id"] == "a"
