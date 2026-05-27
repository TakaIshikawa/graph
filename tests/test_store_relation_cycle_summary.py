import pytest

from graph.store import summarize_relation_cycles


def test_relation_cycles_detects_without_duplicate_rotations():
    report = summarize_relation_cycles(
        [
            {"source_id": "a", "target_id": "b", "relation_type": "supports"},
            {"source_id": "b", "target_id": "a", "relation_type": "supports"},
            {"source_id": "b", "target_id": "c", "relation_type": "references"},
            {"source_id": "c", "target_id": "a", "relation_type": "references"},
            {"source_id": "a", "target_id": "b", "relation_type": "supports"},
        ],
        max_depth=3,
    )

    assert report["cycle_count"] == 2
    assert report["node_count_in_cycles"] == 3
    assert report["cycle_samples"][0]["nodes"] == ["a", "b"]


def test_relation_cycles_validates_max_depth():
    with pytest.raises(ValueError):
        summarize_relation_cycles([], max_depth=0)
    with pytest.raises(ValueError):
        summarize_relation_cycles([], max_depth=1.5)  # type: ignore[arg-type]


def test_relation_cycle_type_counts_reflect_sampled_cycles():
    report = summarize_relation_cycles(
        [
            {"from_unit_id": "x", "to_unit_id": "y", "type": "contains"},
            {"from_unit_id": "y", "to_unit_id": "x", "type": "references"},
        ]
    )

    assert report["relation_type_counts"] == {"contains": 1, "references": 1}
