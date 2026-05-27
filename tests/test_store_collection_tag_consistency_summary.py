from __future__ import annotations

from graph.store.collection_tag_consistency_summary import summarize_collection_tag_consistency


def test_collection_tag_consistency_reports_overlap_and_differences():
    summary = summarize_collection_tag_consistency(
        [{"id": "c1", "tags": ["alpha", "beta"], "member_ids": ["u1", "u2"]}],
        [{"id": "u1", "tags": ["alpha", "gamma"]}, {"id": "u2", "tags": ["alpha"]}],
    )

    assert summary == {
        "total_collections": 1,
        "rows": [
            {
                "collection_id": "c1",
                "member_count": 2,
                "consistency_ratio": 0.5,
                "missing_collection_tags": ["beta"],
                "member_only_tags": ["gamma"],
            }
        ],
    }


def test_collection_tag_consistency_handles_empty_collections_and_nested_members():
    summary = summarize_collection_tag_consistency(
        [{"id": "c1", "metadata": {"items": [{"unit_id": "u1"}]}} , {"id": "c2", "tags": []}],
        [{"id": "u1"}],
    )

    assert [row["consistency_ratio"] for row in summary["rows"]] == [1.0, 1.0]
