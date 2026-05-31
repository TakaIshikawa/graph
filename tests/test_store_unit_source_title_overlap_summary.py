from __future__ import annotations

from graph.store.unit_source_title_overlap_summary import summarize_unit_source_title_overlap


def test_source_title_overlap_buckets_and_high_overlap_examples():
    summary = summarize_unit_source_title_overlap(
        [
            {"id": "u1", "title": "Example News", "metadata": {"site_name": "Example News"}},
            {"id": "u2", "title": "Example News Analysis", "metadata": {"source": "Example"}},
            {"id": "u3", "title": "Example field note", "metadata": {"provider": "Example News"}},
        ]
    )

    assert summary["total_units"] == 3
    assert summary["overlap_buckets"] == [{"bucket": "high", "count": 1}, {"bucket": "low", "count": 2}]
    assert summary["examples"]["high_overlap"] == [{"unit_id": "u1", "title": "Example News", "source": "Example News", "overlap": 1.0}]


def test_source_title_overlap_handles_missing_title_or_source():
    summary = summarize_unit_source_title_overlap([{"id": "u1", "metadata": {"source": "Reader"}}, {"id": "u2", "title": "Untitled"}])

    assert summary["overlap_buckets"] == [{"bucket": "missing_source", "count": 1}, {"bucket": "missing_title", "count": 1}]
    assert summary["examples"]["high_overlap"] == []
