from __future__ import annotations

from graph.store.unit_duplicate_content_hash_summary import summarize_unit_duplicate_content_hashes


def test_unit_duplicate_content_hash_summary_groups_normalized_content():
    summary = summarize_unit_duplicate_content_hashes(
        [
            {"id": "b", "title": "Beta", "content": " Same\ncontent "},
            {"id": "a", "title": "Alpha", "content": "same content"},
            {"id": "c", "content": "   "},
            {"id": "d", "content": "Different"},
        ]
    )

    assert summary["total_units"] == 4
    assert summary["duplicate_group_count"] == 1
    assert summary["duplicate_groups"][0]["unit_ids"] == ["a", "b"]
    assert summary["duplicate_groups"][0]["count"] == 2
    assert summary["duplicate_groups"][0]["title_samples"] == ["Alpha", "Beta"]
