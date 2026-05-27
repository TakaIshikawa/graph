from __future__ import annotations

from graph.store.relation_reciprocity_summary import summarize_relation_reciprocity


def test_relation_reciprocity_counts_pairs_one_way_self_loops_and_duplicates():
    summary = summarize_relation_reciprocity(
        [
            {"relation": "related", "source": "manual", "from_unit_id": "a", "to_unit_id": "b"},
            {"relation": "related", "source": "manual", "from_unit_id": "b", "to_unit_id": "a"},
            {"relation": "related", "source": "manual", "from_unit_id": "a", "to_unit_id": "b"},
            {"relation": "related", "source": "manual", "from_unit_id": "a", "to_unit_id": "c"},
            {"relation": "related", "source": "manual", "from_unit_id": "d", "to_unit_id": "d"},
            {"relation": "depends_on", "source": "import", "from_unit_id": "x", "to_unit_id": "y"},
        ]
    )

    assert summary["rows"] == [
        {
            "relation": "depends_on",
            "source": "import",
            "edge_count": 1,
            "reciprocal_edge_count": 0,
            "one_way_edge_count": 1,
            "self_loop_count": 0,
            "reciprocal_ratio": 0.0,
        },
        {
            "relation": "related",
            "source": "manual",
            "edge_count": 5,
            "reciprocal_edge_count": 2,
            "one_way_edge_count": 1,
            "self_loop_count": 1,
            "reciprocal_ratio": 0.6667,
        },
    ]


def test_relation_reciprocity_handles_missing_endpoints_and_empty_input():
    assert summarize_relation_reciprocity([]) == {"rows": [], "row_count": 0, "edge_count": 0}

    summary = summarize_relation_reciprocity([{"relation": "related"}, {"relation": "related", "from_unit_id": "a"}])

    assert summary["rows"] == [
        {
            "relation": "related",
            "source": None,
            "edge_count": 2,
            "reciprocal_edge_count": 0,
            "one_way_edge_count": 0,
            "self_loop_count": 0,
            "reciprocal_ratio": 0.0,
        }
    ]
