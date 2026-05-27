from __future__ import annotations

from graph.rag.context_source_overlap import analyze_context_source_overlap


def test_context_source_overlap_normalizes_urls_and_tracking_parameters():
    report = analyze_context_source_overlap(
        [
            {"id": "a", "url": "https://example.com/page?utm_source=x&id=1#top"},
            {"id": "b", "url": "https://www.example.com/page?id=1&utm_medium=y"},
            {"id": "c", "url": "https://other.test/story"},
        ]
    )

    assert report["overlap_groups"] == [{"type": "url", "value": "https://example.com/page?id=1", "item_ids": ["a", "b"], "count": 2}]
    assert report["repeated_domain_counts"] == [{"domain": "example.com", "count": 2}]
    assert report["redundancy_ratio"] == 0.6667
    assert report["risk_level"] == "high"


def test_context_source_overlap_groups_repeated_titles_and_handles_missing_metadata():
    report = analyze_context_source_overlap(
        [
            {"id": "1", "title": "Same Story"},
            {"id": "2", "title": "same story"},
            "plain context without metadata",
        ]
    )

    assert report["overlap_groups"] == [{"type": "title", "value": "same story", "item_ids": ["1", "2"], "count": 2}]
    assert report["repeated_domain_counts"] == []
    assert report["risk_level"] == "high"
