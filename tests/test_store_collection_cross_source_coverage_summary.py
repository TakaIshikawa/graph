from __future__ import annotations

from graph.store.collection_cross_source_coverage_summary import summarize_collection_cross_source_coverage


def test_collection_cross_source_coverage_counts_sources_and_missing_members():
    summary = summarize_collection_cross_source_coverage(
        [
            {"id": "c1", "source": "collections", "type": "folder", "member_ids": ["u1", "u2", "missing"]},
            {"id": "c2", "source": "collections", "type": "folder", "members": [{"unit_id": "u1"}, {"id": "u3"}]},
        ],
        [
            {"id": "u1", "source_project": "notes"},
            {"id": "u2", "source_project": "web"},
            {"id": "u3", "source_project": "notes"},
        ],
    )

    assert summary["total_collections"] == 2
    assert summary["cross_source_collection_count"] == 1
    first = summary["collection_summaries"][0]
    assert first["collection_id"] == "c1"
    assert first["missing_member_count"] == 1
    assert first["dominant_source"] == "notes"
    assert first["is_cross_source"] is True
    assert summary["collection_summaries"][1]["is_single_source"] is True


def test_collection_cross_source_coverage_orders_rows_and_tie_breaks_dominant_source():
    summary = summarize_collection_cross_source_coverage(
        [
            {"id": "b", "source": "z", "type": "x", "member_ids": ["u2", "u1"]},
            {"id": "a", "source": "a", "type": "x", "metadata": {"items": ["missing"]}},
        ],
        [{"id": "u1", "source_project": "Beta"}, {"id": "u2", "source_project": "alpha"}],
    )

    assert [row["collection_id"] for row in summary["collection_summaries"]] == ["a", "b"]
    assert summary["collection_summaries"][1]["dominant_source"] == "alpha"
