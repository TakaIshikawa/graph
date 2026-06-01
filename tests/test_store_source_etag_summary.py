from __future__ import annotations

from graph.store import summarize_source_etags


def test_source_etag_summary_counts_presence_strength_and_duplicates():
    summary = summarize_source_etags(
        [
            {"source_id": "a", "etag": '"same"'},
            {"source_id": "b", "metadata": {"headers": {"ETag": 'W/"weak"'}}},
            {"source_id": "c", "metadata": {"etag": '"same"'}},
            {"source_id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_etag"] == 3
    assert summary["sources_missing_etag"] == 1
    assert summary["weak_etag_count"] == 1
    assert summary["strong_etag_count"] == 2
    assert summary["duplicate_etags"] == [{"etag": '"same"', "count": 2, "source_ids": ["a", "c"]}]
    assert summary["examples"][0] == {"source_id": "a", "etag": '"same"'}


def test_source_etag_summary_prefers_top_level_values():
    summary = summarize_source_etags([{"source_id": "s", "etag": '"top"', "metadata": {"etag": '"meta"'}}])

    assert summary["examples"] == [{"source_id": "s", "etag": '"top"'}]
