from __future__ import annotations

from graph.store.collection_member_tag_density_summary import summarize_collection_member_tag_density


def test_collection_member_tag_density_handles_empty_sparse_and_dense_collections():
    summary = summarize_collection_member_tag_density(
        [
            {"id": "empty", "members": []},
            {"id": "mixed", "members": [{"id": "u1", "tags": ["a", "a", "b"]}, {"id": "u2"}, {"id": "u3", "metadata": {"tags": "c, d"}}]},
            {"id": "dense", "members": [{"id": "u4", "tags": ["a", "b"]}, {"id": "u5", "tags": ["c", "d"]}]},
        ],
        minimum_average_tags=1.5,
    )

    assert summary["collections"] == [
        {"collection_id": "dense", "total_members": 2, "tagged_members": 2, "untagged_members": 0, "average_tags_per_member": 2.0, "max_tags_on_member": 2},
        {"collection_id": "empty", "total_members": 0, "tagged_members": 0, "untagged_members": 0, "average_tags_per_member": 0.0, "max_tags_on_member": 0},
        {"collection_id": "mixed", "total_members": 3, "tagged_members": 2, "untagged_members": 1, "average_tags_per_member": 1.33, "max_tags_on_member": 2},
    ]
    assert [row["collection_id"] for row in summary["sparse_collections"]] == ["empty", "mixed"]


def test_collection_member_tag_density_threshold_can_include_or_exclude_average_one():
    summary = summarize_collection_member_tag_density([{"id": "c1", "items": [{"tags": ["a"]}, {}]}], minimum_average_tags=0.4)

    assert summary["collections"][0]["average_tags_per_member"] == 0.5
    assert summary["sparse_collections"] == []
