from graph.store import summarize_source_compression_encodings


def test_source_compression_encoding_summary_splits_normalizes_and_counts_missing():
    summary = summarize_source_compression_encodings(
        [
            {"source_id": "a", "Content-Encoding": "GZip, BR"},
            {"source_id": "b", "metadata": {"headers": {"content-encoding": "deflate"}}},
            {"source_id": "c"},
        ]
    )

    assert summary["encoding_counts"] == {"br": 1, "deflate": 1, "gzip": 1}
    assert summary["missing_encoding_count"] == 1
    assert summary["samples"] == [
        {"source_id": "a", "encodings": ["gzip", "br"]},
        {"source_id": "b", "encodings": ["deflate"]},
    ]
