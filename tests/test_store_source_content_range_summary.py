from graph.store import summarize_source_content_ranges


def test_content_range_summary_parses_ranges_unsatisfied_and_totals():
    summary = summarize_source_content_ranges(
        [
            {"id": "b", "Content-Range": "bytes 0-99/1000"},
            {"id": "a", "metadata": {"response_headers": {"Content_Range": "bytes */1000"}}},
            {"id": "c", "headers": {"content-range": "items 0-4/*"}},
            {"id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_content_range"] == 3
    assert summary["missing_content_range_count"] == 1
    assert summary["unit_counts"] == {"bytes": 2, "items": 1}
    assert summary["unsatisfied_count"] == 1
    assert summary["unknown_total_count"] == 1
    assert summary["complete_count"] == 0
    assert summary["samples"][1] == {"source_id": "b", "raw": "bytes 0-99/1000", "unit": "bytes", "start": 0, "end": 99, "total": 1000}


def test_content_range_summary_counts_complete_and_tolerates_malformed_values():
    summary = summarize_source_content_ranges(
        [
            {"id": "complete", "Content-Range": "bytes 0-999/1000"},
            {"id": "bad", "Content-Range": "not a range"},
        ],
        sample_limit=5,
    )

    assert summary["complete_count"] == 1
    assert summary["malformed_count"] == 1
    assert summary["unknown_total_count"] == 1
    assert summary["unit_counts"] == {"bytes": 1, "unknown": 1}
    assert {"source_id": "bad", "raw": "not a range", "unit": "unknown", "malformed": True} in summary["samples"]
