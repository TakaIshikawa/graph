from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def _unit(unit_id: str, metadata: dict) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project="alpha",
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        metadata=metadata,
    )


def test_metadata_enum_candidates_empty_store_returns_empty(store: Store):
    assert store.metadata_enum_candidates() == []


def test_metadata_enum_candidates_flattens_nested_scalar_paths(store: Store):
    store.insert_unit(_unit("a", {"status": "open", "review": {"state": "draft"}}))
    store.insert_unit(_unit("b", {"status": "closed", "review": {"state": "draft"}}))
    store.insert_unit(_unit("c", {"status": "open", "review": {"state": "done"}}))

    rows = store.metadata_enum_candidates(max_distinct_values=2, min_occurrence_count=2)

    by_key = {row["key"]: row for row in rows}
    assert by_key["review.state"] == {
        "key": "review.state",
        "occurrence_count": 3,
        "distinct_value_count": 2,
        "coverage_ratio": 1.0,
        "value_types": ["string"],
        "top_values": [
            {"value": "draft", "value_type": "string", "count": 2},
            {"value": "done", "value_type": "string", "count": 1},
        ],
    }
    assert by_key["status"]["top_values"] == [
        {"value": "open", "value_type": "string", "count": 2},
        {"value": "closed", "value_type": "string", "count": 1},
    ]


def test_metadata_enum_candidates_ignores_non_scalar_values_and_reports_coverage(store: Store):
    store.insert_unit(_unit("a", {"kind": "note", "labels": ["one"], "owner": {"id": 1}}))
    store.insert_unit(_unit("b", {"kind": "task", "labels": ["two"], "owner": {"id": 2}}))
    store.insert_unit(_unit("c", {"labels": ["three"], "owner": {"id": 3}}))

    rows = store.metadata_enum_candidates(max_distinct_values=3, min_occurrence_count=1)
    by_key = {row["key"]: row for row in rows}

    assert "labels" not in by_key
    assert by_key["kind"]["coverage_ratio"] == pytest.approx(2 / 3)
    assert by_key["owner.id"]["distinct_value_count"] == 3


def test_metadata_enum_candidates_filters_and_limits(store: Store):
    store.insert_unit(_unit("a", {"metrics": {"priority": "high", "state": "open"}}))
    store.insert_unit(_unit("b", {"metrics": {"priority": "low", "state": "open"}}))
    store.insert_unit(_unit("c", {"metrics": {"priority": "high", "state": "closed"}}))

    assert [row["key"] for row in store.metadata_enum_candidates(prefix="metrics", limit=1)] == [
        "metrics.priority"
    ]
    assert store.metadata_enum_candidates(max_distinct_values=1) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_distinct_values": 0}, "max_distinct_values must be a positive integer"),
        ({"max_distinct_values": True}, "max_distinct_values must be a positive integer"),
        ({"min_occurrence_count": 0}, "min_occurrence_count must be a positive integer"),
        ({"limit": 0}, "limit must be a positive integer or None"),
    ],
)
def test_metadata_enum_candidates_validates_options(store: Store, kwargs, message):
    with pytest.raises(ValueError, match=message):
        store.metadata_enum_candidates(**kwargs)
