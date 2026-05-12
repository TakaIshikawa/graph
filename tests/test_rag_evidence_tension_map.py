from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag import map_evidence_tensions


@dataclass
class Result:
    id: str
    source_project: str
    metadata: dict


def test_evidence_tension_map_detects_default_field_disagreements():
    rows = map_evidence_tensions(
        [
            {"id": "a", "source_project": "paper", "metadata": {"stance": "Supports"}},
            {"id": "b", "source_project": "notes", "metadata": {"stance": "disputes"}},
            {"id": "c", "source_project": "paper", "metadata": {"stance": "supports"}},
        ]
    )

    assert rows == [
        {
            "field": "stance",
            "value_counts": [{"value": "supports", "count": 2}, {"value": "disputes", "count": 1}],
            "representative_results": {"disputes": ["b"], "supports": ["a", "c"]},
            "source_projects": ["notes", "paper"],
            "tension_score": 1.0,
        }
    ]


def test_evidence_tension_map_supports_custom_dotted_fields_and_objects():
    rows = map_evidence_tensions(
        [
            Result("a", "reviews", {"assessment": {"rating": "high"}}),
            Result("b", "reviews", {"assessment": {"rating": "low"}}),
            Result("c", "reviews", {"assessment": {"rating": "high"}}),
        ],
        fields=["metadata.assessment.rating"],
    )

    assert rows[0]["field"] == "metadata.assessment.rating"
    assert rows[0]["value_counts"] == [{"value": "high", "count": 2}, {"value": "low", "count": 1}]
    assert rows[0]["representative_results"] == {"high": ["a", "c"], "low": ["b"]}


def test_evidence_tension_map_omits_fields_with_fewer_than_two_values():
    assert map_evidence_tensions(
        [
            {"id": "a", "metadata": {"answer": "yes"}},
            {"id": "b", "metadata": {"answer": "YES"}},
        ]
    ) == []


def test_evidence_tension_map_limit_truncates_sorted_rows():
    rows = map_evidence_tensions(
        [
            {"id": "a", "metadata": {"status": "open", "sentiment": "positive"}},
            {"id": "b", "metadata": {"status": "closed", "sentiment": "negative"}},
        ],
        limit=1,
    )

    assert len(rows) == 1
    assert rows[0]["field"] == "sentiment"


@pytest.mark.parametrize("fields", [[""], [None]])
def test_evidence_tension_map_validates_field_names(fields):
    with pytest.raises(ValueError, match="fields must contain non-empty field names"):
        map_evidence_tensions([], fields=fields)


@pytest.mark.parametrize("limit", [-1, True, "1"])
def test_evidence_tension_map_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        map_evidence_tensions([], limit=limit)
