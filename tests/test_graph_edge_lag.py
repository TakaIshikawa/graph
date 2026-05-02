from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _unit(
    unit_id: str,
    title: str,
    *,
    created_at: str,
    updated_at: str,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        created_at=_dt(created_at),
        updated_at=_dt(updated_at),
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
    )


def _insert_lag_graph(store: Store) -> GraphService:
    for unit in [
        _unit(
            "unit-a",
            "Alpha",
            created_at="2026-01-01T10:00:00+00:00",
            updated_at="2026-02-10T10:00:00+00:00",
        ),
        _unit(
            "unit-b",
            "Beta",
            created_at="2026-01-06T10:00:00+00:00",
            updated_at="2026-02-08T10:00:00+00:00",
        ),
        _unit(
            "unit-c",
            "Gamma",
            created_at="2026-01-03T10:00:00+00:00",
            updated_at="2026-02-08T10:00:00+00:00",
        ),
        _unit(
            "unit-d",
            "Delta",
            created_at="2026-01-10T10:00:00+00:00",
            updated_at="2026-02-20T10:00:00+00:00",
        ),
        _unit(
            "unit-e",
            "Epsilon",
            created_at="2026-01-01T23:00:00+00:00",
            updated_at="2026-02-12T10:00:00+00:00",
        ),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-a-b", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
        _edge("edge-b-c", "unit-b", "unit-c", EdgeRelation.BUILDS_ON),
        _edge("edge-a-c", "unit-a", "unit-c", EdgeRelation.REFERENCES),
        _edge("edge-c-d", "unit-c", "unit-d", EdgeRelation.REFERENCES),
        _edge("edge-a-e", "unit-a", "unit-e", EdgeRelation.INSPIRES),
    ]:
        store.insert_edge(edge)

    service = GraphService(store)
    service.rebuild()
    service.G.nodes["unit-d"]["created_at"] = ""
    return service


def test_analyze_edge_lag_groups_relations_and_reports_missing_dates(store: Store):
    result = _insert_lag_graph(store).analyze_edge_lag(example_limit=2)

    assert result["field"] == "created_at"
    assert result["node_count"] == 5
    assert result["edge_count"] == 5
    assert result["relation_count"] == 3

    assert result["relations"] == [
        {
            "relation": "builds_on",
            "edge_count": 2,
            "dated_edge_count": 2,
            "missing_date_count": 0,
            "min_days": -3,
            "max_days": 5,
            "average_days": 1.0,
            "median_days": 1.0,
            "forward_count": 1,
            "backward_count": 1,
            "examples": [
                {
                    "from_unit_id": "unit-a",
                    "to_unit_id": "unit-b",
                    "relation": "builds_on",
                    "lag_days": 5,
                },
                {
                    "from_unit_id": "unit-b",
                    "to_unit_id": "unit-c",
                    "relation": "builds_on",
                    "lag_days": -3,
                },
            ],
        },
        {
            "relation": "references",
            "edge_count": 2,
            "dated_edge_count": 1,
            "missing_date_count": 1,
            "min_days": 2,
            "max_days": 2,
            "average_days": 2.0,
            "median_days": 2,
            "forward_count": 1,
            "backward_count": 0,
            "examples": [
                {
                    "from_unit_id": "unit-a",
                    "to_unit_id": "unit-c",
                    "relation": "references",
                    "lag_days": 2,
                }
            ],
        },
        {
            "relation": "inspires",
            "edge_count": 1,
            "dated_edge_count": 1,
            "missing_date_count": 0,
            "min_days": 0,
            "max_days": 0,
            "average_days": 0.0,
            "median_days": 0,
            "forward_count": 1,
            "backward_count": 0,
            "examples": [
                {
                    "from_unit_id": "unit-a",
                    "to_unit_id": "unit-e",
                    "relation": "inspires",
                    "lag_days": 0,
                }
            ],
        },
    ]


def test_analyze_edge_lag_uses_updated_at_and_limits_examples(store: Store):
    result = _insert_lag_graph(store).analyze_edge_lag(
        field="updated_at",
        example_limit=1,
    )

    builds_on = result["relations"][0]

    assert result["field"] == "updated_at"
    assert builds_on["relation"] == "builds_on"
    assert builds_on["min_days"] == -2
    assert builds_on["max_days"] == 0
    assert builds_on["average_days"] == -1.0
    assert builds_on["median_days"] == -1.0
    assert builds_on["forward_count"] == 1
    assert builds_on["backward_count"] == 1
    assert builds_on["examples"] == [
        {
            "from_unit_id": "unit-a",
            "to_unit_id": "unit-b",
            "relation": "builds_on",
            "lag_days": -2,
        }
    ]


def test_analyze_edge_lag_handles_empty_graph(store: Store):
    assert GraphService(store).analyze_edge_lag() == {
        "field": "created_at",
        "node_count": 0,
        "edge_count": 0,
        "relation_count": 0,
        "relations": [],
    }

