from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.result_entity_coverage import analyze_result_entity_coverage


@dataclass
class Result:
    id: str
    metadata: dict
    source_project: str


def test_entity_coverage_counts_payload_metadata_objects_and_tuples():
    rows = analyze_result_entity_coverage(
        [
            {"id": "a", "entities": ["Acme", "Battery"], "source_project": "notes"},
            ({"id": "b", "metadata": {"people": [{"name": "Ada"}], "entities": ["acme"]}, "source_project": "web"}, 0.8),
            Result("c", {"authors": ["Ada"], "projects": ["Grid"]}, "notes"),
        ]
    )

    assert rows[:3] == [
        {
            "entity": "Acme",
            "count": 2,
            "result_ids": ["a", "b"],
            "source_projects": ["notes", "web"],
            "coverage_ratio": 0.666667,
        },
        {
            "entity": "Ada",
            "count": 2,
            "result_ids": ["b", "c"],
            "source_projects": ["notes", "web"],
            "coverage_ratio": 0.666667,
        },
        {
            "entity": "Battery",
            "count": 1,
            "result_ids": ["a"],
            "source_projects": ["notes"],
            "coverage_ratio": 0.333333,
        },
    ]


def test_custom_keys_and_limit_are_stable():
    rows = analyze_result_entity_coverage(
        [
            {"id": "1", "metadata": {"teams": ["Zeta", "Alpha"]}},
            {"id": "2", "teams": ["Alpha"]},
        ],
        entity_keys=["teams"],
        limit=1,
    )

    assert rows == [
        {
            "entity": "Alpha",
            "count": 2,
            "result_ids": ["1", "2"],
            "source_projects": ["unknown"],
            "coverage_ratio": 1.0,
        }
    ]


@pytest.mark.parametrize("limit", [-1, 1.2, True, "2"])
def test_entity_coverage_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        analyze_result_entity_coverage([], limit=limit)
