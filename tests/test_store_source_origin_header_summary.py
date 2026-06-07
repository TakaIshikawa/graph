from graph.store import summarize_source_origin_headers


def test_origin_header_summary_counts_schemes_null_file_missing_and_mismatch():
    summary = summarize_source_origin_headers(
        [
            {"source_id": "a", "url": "https://api.example.test/path", "Origin": "https://app.example.test"},
            {"source_id": "b", "metadata": {"origin": "http://legacy.test"}},
            {"source_id": "c", "headers": {"Origin": "null"}},
            {"source_id": "d", "response_headers": {"origin": "file://local/path"}},
            {"source_id": "e"},
        ]
    )

    assert summary["scheme_counts"] == {"file": 1, "http": 1, "https": 1}
    assert summary["domain_counts"] == {"app.example.test": 1, "legacy.test": 1, "local": 1}
    assert summary["null_origin_count"] == 1
    assert summary["opaque_file_origin_count"] == 2
    assert summary["host_mismatch_count"] == 1
    assert summary["missing_origin_count"] == 1
