from graph.store import summarize_source_trailer_headers


def test_trailer_summary_counts_fields_and_unusual_samples():
    summary = summarize_source_trailer_headers(
        [
            {"source_id": "a", "Trailer": "Digest, Signature"},
            {"source_id": "b", "metadata": {"response_headers": {"TRAILER": "ETag, Custom_Field"}}},
            {"source_id": "c"},
        ],
        sample_limit=1,
    )

    assert summary["sources_with_trailer"] == 2
    assert summary["sources_missing_trailer"] == 1
    assert summary["trailer_field_counts"] == {"custom-field": 1, "digest": 1, "etag": 1, "signature": 1}
    assert summary["digest_trailer_count"] == 1
    assert summary["signature_trailer_count"] == 1
    assert summary["unusual_trailer_samples"] == [{"source_id": "b", "field": "custom-field"}]
