from graph.store import summarize_source_x_download_options


def test_x_download_options_summary_normalizes_values_and_export():
    summary = summarize_source_x_download_options(
        [
            {"id": "a", "headers": {"X-Download-Options": "NoOpen"}},
            {"id": "b", "metadata": {"response_headers": {"x_download_options": "preview"}}},
            {"id": "c"},
        ]
    )

    assert summary["noopen_count"] == 1
    assert summary["other_value_counts"] == {"preview": 1}
    assert summary["missing_x_download_options_count"] == 1
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b"]
