from __future__ import annotations

from graph.store.collection_duplicate_member_summary import summarize_collection_duplicate_members


def test_collection_duplicate_members_detects_duplicates_within_collection():
    summary = summarize_collection_duplicate_members(
        [
            {"source": "docs", "type": "folder", "member_ids": ["a", "a", "b"]},
            {"source": "docs", "type": "folder", "member_ids": ["a", "b"]},
            {"source": "docs", "type": "folder", "members": [{"id": "c"}, {"unit_id": "c"}, {"id": "d"}]},
            {"source": "crm", "type": "list", "metadata": {"items": ["z", "z", "z"]}},
        ]
    )

    rows = {(row["source"], row["collection_type"]): row for row in summary["rows"]}
    assert rows[("docs", "folder")]["duplicate_member_collection_count"] == 2
    assert rows[("docs", "folder")]["duplicate_member_total"] == 2
    assert rows[("docs", "folder")]["sample_duplicate_member_ids"] == ["a", "c"]
    assert rows[("crm", "list")]["duplicate_member_total"] == 2


def test_collection_duplicate_members_orders_groups_and_samples():
    summary = summarize_collection_duplicate_members(
        [{"source": "b", "type": "x", "member_ids": ["2", "2", "1", "1"]}, {"source": "a", "type": "x"}]
    )

    assert [row["source"] for row in summary["rows"]] == ["a", "b"]
    assert summary["rows"][1]["sample_duplicate_member_ids"] == ["1", "2"]
