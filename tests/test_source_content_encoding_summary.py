from __future__ import annotations

from graph.store import summarize_source_content_encodings


def test_content_encoding_summary_normalizes_splits_and_counts_unknowns():
    summary = summarize_source_content_encodings(
        [
            {"id": "a", "headers": {"Content-Encoding": "GZip, BR"}},
            {"id": "b", "metadata": {"response_headers": {"content_encoding": "zstd, X-Custom"}}},
            {"id": "c", "response_headers": {"CONTENT_ENCODING": "identity"}},
            {"id": "d", "content-encoding": "br"},
            {"id": "e"},
            {"id": "f", "metadata": {"content_encoding": "   "}},
        ]
    )

    assert summary["total_sources"] == 6
    assert summary["sources_with_content_encoding"] == 4
    assert summary["missing_content_encoding_count"] == 2
    assert summary["unknown_encoding_count"] == 1
    assert summary["encoding_counts"] == {"br": 2, "gzip": 1, "identity": 1, "x-custom": 1, "zstd": 1}
    assert {"encoding": "x-custom", "count": 1, "known": False, "source_ids": ["b"], "examples": ["zstd, X-Custom"]} in summary["rows"]


def test_content_encoding_summary_sample_limit_and_missing_headers():
    summary = summarize_source_content_encodings(
        [
            {"id": "z", "content_encoding": "deflate"},
            {"id": "a", "headers": {"content-encoding": "Deflate"}},
            {"id": "missing"},
        ],
        sample_limit=1,
    )

    assert summary["missing_content_encoding_count"] == 1
    assert summary["rows"] == [
        {"encoding": "deflate", "count": 2, "known": True, "source_ids": ["z"], "examples": ["deflate"]},
    ]
