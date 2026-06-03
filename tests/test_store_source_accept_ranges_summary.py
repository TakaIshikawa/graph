from graph.store import summarize_source_accept_ranges


def test_accept_ranges_summary_counts_supported_none_unknown_and_missing_values():
    summary = summarize_source_accept_ranges(
        [
            {"id": "b", "Accept-Ranges": "bytes"},
            {"id": "n", "metadata": {"accept_ranges": "NONE"}},
            {"id": "u", "headers": {"accept-ranges": "items"}},
            {"id": "d", "metadata": {"response_headers": {"Accept_Ranges": "Bytes"}}},
            {"id": "e", "response_headers": {"Accept-Ranges": ""}},
            {"id": "m"},
        ]
    )

    assert summary["total_sources"] == 6
    assert summary["sources_with_accept_ranges"] == 4
    assert summary["missing_header_count"] == 1
    assert summary["empty_value_count"] == 1
    assert summary["value_counts"] == {"bytes": 2, "items": 1, "none": 1}
    assert summary["byte_range_source_count"] == 2
    assert summary["none_source_count"] == 1
    assert summary["unknown_value_count"] == 1
    assert summary["samples"] == [
        {"source_id": "b", "accept_ranges": "bytes"},
        {"source_id": "d", "accept_ranges": "bytes"},
    ]


def test_accept_ranges_summary_respects_sample_limit_while_counting():
    summary = summarize_source_accept_ranges(
        [{"id": "b", "Accept-Ranges": "bytes"}, {"id": "a", "Accept-Ranges": "bytes"}],
        sample_limit=1,
    )

    assert summary["byte_range_source_count"] == 2
    assert summary["samples"] == [{"source_id": "b", "accept_ranges": "bytes"}]
