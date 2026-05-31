from __future__ import annotations

from graph.store.source_content_type_summary import summarize_source_content_types


def test_source_content_types_normalizes_metadata_content_type_and_parameters():
    summary = summarize_source_content_types(
        [
            {"id": "html", "metadata": {"content_type": "Text/HTML; charset=utf-8"}},
            {"id": "json", "metadata": {"mime_type": "application/json"}},
            {"id": "fallback", "content_type": "IMAGE/PNG"},
        ]
    )

    assert summary["total_sources"] == 3
    assert summary["missing_content_type_count"] == 0
    assert summary["invalid_content_type_count"] == 0
    assert summary["media_type_counts"] == {"application/json": 1, "image/png": 1, "text/html": 1}
    assert summary["top_level_type_counts"] == {"application": 1, "image": 1, "text": 1}
    assert summary["source_counts"] == {"fallback": 1, "html": 1, "json": 1}


def test_source_content_types_reports_missing_and_malformed_values_separately():
    summary = summarize_source_content_types(
        [
            {"id": "missing", "metadata": {}},
            {"id": "blank", "metadata": {"content_type": " "}},
            {"id": "bad", "metadata": {"content_type": "html"}},
            {"id": "also-bad", "metadata": {"mime_type": "text html"}},
        ]
    )

    assert summary["missing_content_type_count"] == 2
    assert summary["invalid_content_type_count"] == 2
    assert summary["media_type_counts"] == {}
    assert [sample["bucket"] for sample in summary["samples"]] == ["invalid", "invalid", "missing", "missing"]


def test_source_content_types_honors_sample_limit_and_empty_input():
    assert summarize_source_content_types([], sample_limit=-1) == {
        "total_sources": 0,
        "missing_content_type_count": 0,
        "invalid_content_type_count": 0,
        "media_type_counts": {},
        "top_level_type_counts": {},
        "source_counts": {},
        "samples": [],
    }

    summary = summarize_source_content_types([{"id": "a", "content_type": "text/plain"}, {"id": "b"}], sample_limit=1)
    assert len(summary["samples"]) == 1
