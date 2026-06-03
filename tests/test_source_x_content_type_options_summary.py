from graph.store.source_x_content_type_options_summary import summarize_source_x_content_type_options


def test_x_content_type_options_summary_counts_nosniff_case_insensitively():
    summary = summarize_source_x_content_type_options(
        [
            {"source_id": "b", "metadata": {"headers": {"x-content-type-options": "NoSniff"}}},
            {"source_id": "a", "X-Content-Type-Options": "nosniff"},
            {"source_id": "c"},
        ]
    )

    assert summary["value_counts"] == {"nosniff": 2}
    assert summary["missing_header_count"] == 1
    assert summary["source_ids"] == ["a", "b"]


def test_x_content_type_options_summary_reports_unexpected_values():
    summary = summarize_source_x_content_type_options(
        [{"source_id": "u", "response_headers": {"x_content_type_options": "maybe"}}]
    )

    assert summary["unexpected_value_count"] == 1
    assert summary["unexpected_values"] == [{"value": "maybe", "count": 1, "source_ids": ["u"]}]
