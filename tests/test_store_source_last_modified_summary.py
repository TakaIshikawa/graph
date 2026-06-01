from graph.store import summarize_source_last_modified


def test_source_last_modified_summary_counts_parseable_invalid_and_missing():
    summary = summarize_source_last_modified(
        [
            {"source_id": "a", "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT"},
            {"source_id": "b", "metadata": {"headers": {"last-modified": "2020-01-01T00:00:00Z"}}},
            {"source_id": "c", "metadata": {"last_modified": "not a date"}},
            {"source_id": "d"},
        ]
    )

    assert summary["present_count"] == 3
    assert summary["missing_count"] == 1
    assert summary["parseable_count"] == 2
    assert summary["invalid_count"] == 1
    assert summary["oldest"] == "2015-10-21T07:28:00+00:00"
    assert summary["newest"] == "2020-01-01T00:00:00+00:00"
    assert [sample["source_id"] for sample in summary["samples"]] == ["c", "d"]
