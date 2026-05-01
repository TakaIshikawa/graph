"""Tests for edge bridge analysis."""

from __future__ import annotations

import os
import tempfile

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _unit(unit_id: str, title: str, tags: list[str] | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
    weight: float = 1.0,
    source: EdgeSource = EdgeSource.INFERRED,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=weight,
        source=source,
    )


@pytest.fixture
def bridge_store(store: Store):
    for unit in [
        _unit("unit-alpha", "Alpha"),
        _unit("unit-beta", "Beta"),
        _unit("unit-gamma", "Gamma"),
        _unit("unit-delta", "Delta"),
        _unit("unit-epsilon", "Epsilon"),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-alpha-beta", "unit-alpha", "unit-beta"),
        _edge("edge-beta-gamma", "unit-beta", "unit-gamma"),
        _edge("edge-gamma-alpha", "unit-gamma", "unit-alpha"),
        _edge(
            "edge-gamma-delta",
            "unit-gamma",
            "unit-delta",
            EdgeRelation.BUILDS_ON,
            0.7,
            EdgeSource.MANUAL,
        ),
        _edge(
            "edge-delta-gamma",
            "unit-delta",
            "unit-gamma",
            EdgeRelation.REFERENCES,
            0.4,
            EdgeSource.SOURCE,
        ),
        _edge(
            "edge-delta-epsilon",
            "unit-delta",
            "unit-epsilon",
            EdgeRelation.INSPIRES,
            0.8,
            EdgeSource.INFERRED,
        ),
    ]:
        store.insert_edge(edge)

    return store


def test_analyze_edge_bridges_returns_payloads_with_impact(bridge_store: Store):
    result = GraphService(bridge_store).analyze_edge_bridges()

    assert [record["unit_ids"] for record in result] == [
        ["unit-delta", "unit-gamma"],
        ["unit-delta", "unit-epsilon"],
    ]

    first = result[0]
    assert first["endpoints"] == [
        {
            "id": "unit-delta",
            "source_project": "max",
            "source_id": "unit-delta",
            "source_entity_type": "insight",
            "title": "Delta",
            "content_type": "insight",
        },
        {
            "id": "unit-gamma",
            "source_project": "max",
            "source_id": "unit-gamma",
            "source_entity_type": "insight",
            "title": "Gamma",
            "content_type": "insight",
        },
    ]
    assert first["relations"] == ["builds_on", "references"]
    assert first["sources"] == ["manual", "source"]
    assert first["weight"] == 0.7
    assert first["total_weight"] == 1.1
    assert [
        {
            "id": edge["id"],
            "from_unit_id": edge["from_unit_id"],
            "to_unit_id": edge["to_unit_id"],
            "relation": edge["relation"],
            "weight": edge["weight"],
            "source": edge["source"],
        }
        for edge in first["edges"]
    ] == [
        {
            "id": "edge-gamma-delta",
            "from_unit_id": "unit-gamma",
            "to_unit_id": "unit-delta",
            "relation": "builds_on",
            "weight": 0.7,
            "source": "manual",
        },
        {
            "id": "edge-delta-gamma",
            "from_unit_id": "unit-delta",
            "to_unit_id": "unit-gamma",
            "relation": "references",
            "weight": 0.4,
            "source": "source",
        },
    ]
    assert first["impact"] == {
        "component_count_before": 1,
        "component_count_after": 2,
        "original_component_size": 5,
        "endpoint_component_sizes": {
            "unit-delta": 2,
            "unit-gamma": 3,
        },
        "smaller_component_size": 2,
        "larger_component_size": 3,
    }


def test_analyze_edge_bridges_reports_no_bridges_for_cycle(store: Store):
    for unit in [
        _unit("unit-alpha", "Alpha"),
        _unit("unit-beta", "Beta"),
        _unit("unit-gamma", "Gamma"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-alpha-beta", "unit-alpha", "unit-beta"),
        _edge("edge-beta-gamma", "unit-beta", "unit-gamma"),
        _edge("edge-gamma-alpha", "unit-gamma", "unit-alpha"),
    ]:
        store.insert_edge(edge)

    assert GraphService(store).analyze_edge_bridges() == []


def test_analyze_edge_bridges_handles_disconnected_components(store: Store):
    for unit in [
        _unit("unit-a", "A"),
        _unit("unit-b", "B"),
        _unit("unit-c", "C"),
        _unit("unit-x", "X"),
        _unit("unit-y", "Y"),
        _unit("unit-z", "Z"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-a-b", "unit-a", "unit-b"),
        _edge("edge-b-c", "unit-b", "unit-c"),
        _edge("edge-c-a", "unit-c", "unit-a"),
        _edge("edge-x-y", "unit-x", "unit-y"),
        _edge("edge-y-z", "unit-y", "unit-z"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_edge_bridges()

    assert [record["unit_ids"] for record in result] == [
        ["unit-x", "unit-y"],
        ["unit-y", "unit-z"],
    ]
    assert [record["impact"]["component_count_before"] for record in result] == [2, 2]
    assert [record["impact"]["component_count_after"] for record in result] == [3, 3]


def test_analyze_edge_bridges_is_deterministic_and_limited(bridge_store: Store):
    service = GraphService(bridge_store)

    first = service.analyze_edge_bridges(limit=1)
    second = GraphService(bridge_store).analyze_edge_bridges(limit=1)

    assert first == second
    assert [record["unit_ids"] for record in first] == [
        ["unit-delta", "unit-gamma"]
    ]


@pytest.mark.parametrize("limit", [0, "0"])
def test_analyze_edge_bridges_accepts_zero_limit(bridge_store: Store, limit):
    assert GraphService(bridge_store).analyze_edge_bridges(limit=limit) == []


@pytest.mark.parametrize("limit", [-1, "many", None])
def test_analyze_edge_bridges_validates_limit(bridge_store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(bridge_store).analyze_edge_bridges(limit=limit)


def test_analyze_edge_bridges_returns_empty_for_empty_graph(store: Store):
    assert GraphService(store).analyze_edge_bridges() == []
