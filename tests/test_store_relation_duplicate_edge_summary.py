from __future__ import annotations

from types import SimpleNamespace

from graph.store import summarize_relation_duplicate_edges


def test_summarize_relation_duplicate_edges_groups_aliases():
    summary = summarize_relation_duplicate_edges(
        [
            {"id": "e1", "source_id": "a", "target_id": "b", "relation_type": "references", "metadata": {"x": 1}},
            SimpleNamespace(id="e2", from_unit_id="a", to_unit_id="b", relation="references", metadata={"y": 2}),
            {"id": "e3", "source_id": "a", "target_id": "c", "relation_type": "references"},
        ]
    )

    assert summary["duplicate_group_count"] == 1
    assert summary["duplicate_edge_count"] == 2
    assert summary["unique_edge_count"] == 1
    assert summary["groups"][0]["edge_ids"] == ["e1", "e2"]
    assert summary["groups"][0]["metadata_key_variation_count"] == 2
