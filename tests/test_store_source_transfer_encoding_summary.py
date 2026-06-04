from graph.store import summarize_source_transfer_encodings


def test_transfer_encoding_summary_splits_normalizes_and_counts_chunked():
    summary = summarize_source_transfer_encodings(
        [
            {"id": "b", "Transfer-Encoding": "gzip, Chunked"},
            {"id": "a", "metadata": {"headers": {"transfer-encoding": "br"}}},
            {"id": "c", "response_headers": {"Transfer_Encoding": "chunked"}},
        ]
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_transfer_encoding"] == 3
    assert summary["missing_transfer_encoding_count"] == 0
    assert summary["encoding_counts"] == {"br": 1, "chunked": 2, "gzip": 1}
    assert summary["chunked_count"] == 2
    assert summary["samples"][1]["transfer_encodings"] == ["gzip", "chunked"]


def test_transfer_encoding_summary_ignores_blank_or_missing_values():
    summary = summarize_source_transfer_encodings(
        [
            {"id": "blank", "Transfer-Encoding": "  "},
            {"id": "missing"},
        ]
    )

    assert summary["sources_with_transfer_encoding"] == 0
    assert summary["missing_transfer_encoding_count"] == 2
    assert summary["samples"] == []
