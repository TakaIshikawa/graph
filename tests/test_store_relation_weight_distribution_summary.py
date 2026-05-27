from __future__ import annotations

from graph.store.relation_weight_distribution_summary import summarize_relation_weight_distribution


def test_relation_weight_distribution_groups_and_summarizes_weights():
    summary = summarize_relation_weight_distribution(
        [
            {"relation": "mentions", "source": "manual", "weight": "0.5"},
            {"relation": "mentions", "source": "manual", "weight": 0},
            {"relation": "mentions", "source": "manual", "weight": "bad"},
            {"relation": "depends_on", "source": "import", "metadata": {"weight": 1}},
        ]
    )

    assert summary["rows"] == [
        {
            "relation": "depends_on",
            "source": "import",
            "edge_count": 1,
            "missing_weight_count": 0,
            "min_weight": 1.0,
            "max_weight": 1.0,
            "average_weight": 1.0,
            "zero_weight_count": 0,
        },
        {
            "relation": "mentions",
            "source": "manual",
            "edge_count": 3,
            "missing_weight_count": 1,
            "min_weight": 0.0,
            "max_weight": 0.5,
            "average_weight": 0.25,
            "zero_weight_count": 1,
        },
    ]


def test_relation_weight_distribution_missing_groups_rounding_and_empty_input():
    assert summarize_relation_weight_distribution([]) == {"rows": [], "row_count": 0, "edge_count": 0}

    summary = summarize_relation_weight_distribution(
        [
            {"weight": 1},
            {"weight": 2},
            {"weight": None},
        ]
    )

    assert summary["rows"] == [
        {
            "relation": None,
            "source": None,
            "edge_count": 3,
            "missing_weight_count": 1,
            "min_weight": 1.0,
            "max_weight": 2.0,
            "average_weight": 1.5,
            "zero_weight_count": 0,
        }
    ]
