from __future__ import annotations

from graph.store.relation_type_coverage_summary import summarize_relation_type_coverage


def test_relation_type_coverage_groups_with_deterministic_ordering():
    summary = summarize_relation_type_coverage(
        [
            {
                "relation": "mentions",
                "from_unit_id": "u1",
                "to_unit_id": "u2",
                "source": "manual",
                "evidence": ["quote"],
                "weight": 0.5,
            },
            {
                "relation": "depends_on",
                "from_unit_id": "u1",
                "to_unit_id": "u3",
                "source": "import",
                "metadata": {"evidence": ["link"], "weight": "0.75"},
            },
            {
                "relation": "mentions",
                "from_unit_id": "u2",
                "to_unit_id": "u2",
                "source": "manual",
                "weight": 1,
            },
        ]
    )

    assert summary == {
        "rows": [
            {
                "relation_type": "depends_on",
                "edge_count": 1,
                "unique_source_unit_count": 1,
                "unique_target_unit_count": 1,
                "distinct_source_count": 1,
                "missing_evidence_count": 0,
                "average_weight": 0.75,
            },
            {
                "relation_type": "mentions",
                "edge_count": 2,
                "unique_source_unit_count": 2,
                "unique_target_unit_count": 1,
                "distinct_source_count": 1,
                "missing_evidence_count": 1,
                "average_weight": 0.75,
            },
        ],
        "row_count": 2,
        "edge_count": 3,
    }


def test_relation_type_coverage_handles_missing_fields_and_empty_input():
    assert summarize_relation_type_coverage([]) == {"rows": [], "row_count": 0, "edge_count": 0}

    summary = summarize_relation_type_coverage([{"type": "related", "weight": "not-a-number"}])

    assert summary["rows"] == [
        {
            "relation_type": "related",
            "edge_count": 1,
            "unique_source_unit_count": 0,
            "unique_target_unit_count": 0,
            "distinct_source_count": 0,
            "missing_evidence_count": 1,
            "average_weight": None,
        }
    ]
