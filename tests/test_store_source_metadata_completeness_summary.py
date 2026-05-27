from __future__ import annotations

from graph.store.source_metadata_completeness_summary import summarize_source_metadata_completeness


def test_source_metadata_completeness_uses_default_keys_and_blank_values():
    summary = summarize_source_metadata_completeness(
        [
            {"id": "s1", "metadata": {"name": "Docs", "url": "https://example.test", "source_type": "html"}},
            {"id": "s2", "metadata": {"name": " ", "url": "https://example.test"}},
            {"id": "s3", "metadata": None},
        ]
    )

    assert summary["source_count"] == 3
    assert summary["complete_source_count"] == 1
    assert summary["incomplete_source_count"] == 2
    assert summary["completeness_ratio"] == 1 / 3
    assert summary["missing_counts_by_key"] == [
        {"key": "name", "count": 2},
        {"key": "source_type", "count": 2},
        {"key": "url", "count": 1},
    ]
    assert summary["samples"] == [
        {"source_id": "s2", "missing_keys": ["name", "source_type"]},
        {"source_id": "s3", "missing_keys": ["name", "url", "source_type"]},
    ]


def test_source_metadata_completeness_supports_custom_required_keys_and_zero_sources():
    assert summarize_source_metadata_completeness([], required_keys=["owner"]) == {
        "source_count": 0,
        "required_keys": ["owner"],
        "complete_source_count": 0,
        "incomplete_source_count": 0,
        "completeness_ratio": 0,
        "missing_counts_by_key": [],
        "samples": [],
    }

    summary = summarize_source_metadata_completeness([{"id": "s1", "owner": "Ada", "metadata": {}}], required_keys=["owner"])
    assert summary["complete_source_count"] == 1
