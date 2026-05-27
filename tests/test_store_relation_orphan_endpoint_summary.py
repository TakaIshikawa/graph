from __future__ import annotations

from types import SimpleNamespace

from graph.store.relation_orphan_endpoint_summary import summarize_relation_orphan_endpoints


def test_relation_orphan_endpoint_summary_detects_missing_sides_and_types():
    summary = summarize_relation_orphan_endpoints(
        [
            {"id": "r1", "relation": "mentions", "from_unit_id": "u1", "to_unit_id": "u-missing"},
            {"id": "r2", "relation": "mentions", "from_unit_id": "u-missing", "to_unit_id": "u2"},
            {"id": "r3", "relation": "blocks", "from_unit_id": "x", "to_unit_id": "y"},
            {"id": "r4", "relation": "mentions", "from_unit_id": "u1", "to_unit_id": "u2"},
        ],
        [{"id": "u1"}, {"unit_id": "u2"}],
    )

    assert summary == {
        "total_relations": 4,
        "orphan_relation_count": 3,
        "missing_endpoint_counts": [
            {"side": "both", "count": 1},
            {"side": "source", "count": 1},
            {"side": "target", "count": 1},
        ],
        "relation_type_counts": [{"relation_type": "blocks", "count": 1}, {"relation_type": "mentions", "count": 2}],
        "rows": [
            {"relation_type": "blocks", "missing_endpoint_side": "both", "count": 1, "example_relation_ids": ["r3"]},
            {"relation_type": "mentions", "missing_endpoint_side": "source", "count": 1, "example_relation_ids": ["r2"]},
            {"relation_type": "mentions", "missing_endpoint_side": "target", "count": 1, "example_relation_ids": ["r1"]},
        ],
    }


def test_relation_orphan_endpoint_summary_supports_objects_and_metadata():
    summary = summarize_relation_orphan_endpoints(
        [SimpleNamespace(edge_id="edge-1", metadata={"relation_type": "links", "source_unit_id": "a", "target_unit_id": "b"})],
        [SimpleNamespace(id="a")],
    )

    assert summary["rows"] == [
        {"relation_type": "links", "missing_endpoint_side": "target", "count": 1, "example_relation_ids": ["edge-1"]}
    ]
