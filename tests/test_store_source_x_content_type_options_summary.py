from graph.store import summarize_source_x_content_type_options


def test_x_content_type_options_summary_counts_values_and_samples_unusual():
    summary = summarize_source_x_content_type_options(
        [
            {"source_id": "a", "X-Content-Type-Options": "nosniff"},
            {"source_id": "b", "metadata": {"headers": {"x-content-type-options": "NoSniff"}}},
            {"source_id": "c", "response_headers": {"x_content_type_options": "maybe"}},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_header"] == 3
    assert summary["value_counts"] == {"maybe": 1, "nosniff": 2}
    assert summary["missing_header_count"] == 1
    assert summary["non_nosniff_count"] == 1
    assert summary["samples"] == [{"source_id": "c", "value": "maybe", "field": "response_headers.x_content_type_options"}]
