from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def test_analyze_connected_components_handles_empty_graph(store: Store):
    assert GraphService(store).analyze_connected_components() == {
        "node_count": 0,
        "edge_count": 0,
        "component_count": 0,
        "isolated_component_count": 0,
        "components": [],
    }


def test_analyze_connected_components_handles_single_isolated_node(store: Store):
    store.insert_unit(_unit("unit-alpha", "Alpha"))

    result = GraphService(store).analyze_connected_components()

    assert result == {
        "node_count": 1,
        "edge_count": 0,
        "component_count": 1,
        "isolated_component_count": 1,
        "components": [
            {
                "component_id": "component-001",
                "unit_ids": ["unit-alpha"],
                "size": 1,
                "representative_titles": ["Alpha"],
                "internal_edge_count": 0,
                "isolated": True,
            }
        ],
    }


def test_analyze_connected_components_orders_multiple_components_deterministically(
    store: Store,
):
    for unit in [
        _unit("unit-delta", "Delta"),
        _unit("unit-alpha", "Alpha"),
        _unit("unit-zeta", "Zeta"),
        _unit("unit-gamma", "Gamma"),
        _unit("unit-beta", "Beta"),
        _unit("unit-isolated", "Isolated"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-alpha-beta", "unit-alpha", "unit-beta"),
        _edge("edge-beta-gamma", "unit-beta", "unit-gamma"),
        _edge("edge-delta-zeta", "unit-delta", "unit-zeta"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_connected_components()

    assert result["node_count"] == 6
    assert result["edge_count"] == 3
    assert result["component_count"] == 3
    assert result["isolated_component_count"] == 1
    assert result["components"] == [
        {
            "component_id": "component-001",
            "unit_ids": ["unit-alpha", "unit-beta", "unit-gamma"],
            "size": 3,
            "representative_titles": ["Beta", "Alpha", "Gamma"],
            "internal_edge_count": 2,
            "isolated": False,
        },
        {
            "component_id": "component-002",
            "unit_ids": ["unit-delta", "unit-zeta"],
            "size": 2,
            "representative_titles": ["Delta", "Zeta"],
            "internal_edge_count": 1,
            "isolated": False,
        },
        {
            "component_id": "component-003",
            "unit_ids": ["unit-isolated"],
            "size": 1,
            "representative_titles": ["Isolated"],
            "internal_edge_count": 0,
            "isolated": True,
        },
    ]


def test_analyze_connected_components_treats_directed_edges_as_connected(
    store: Store,
):
    for unit in [
        _unit("unit-alpha", "Alpha"),
        _unit("unit-beta", "Beta"),
        _unit("unit-gamma", "Gamma"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-beta-alpha", "unit-beta", "unit-alpha"),
        _edge("edge-beta-gamma", "unit-beta", "unit-gamma"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_connected_components(
        representative_limit=2,
    )

    assert result["component_count"] == 1
    assert result["isolated_component_count"] == 0
    assert result["components"] == [
        {
            "component_id": "component-001",
            "unit_ids": ["unit-alpha", "unit-beta", "unit-gamma"],
            "size": 3,
            "representative_titles": ["Beta", "Alpha"],
            "internal_edge_count": 2,
            "isolated": False,
        }
    ]


def test_analyze_connected_components_limit_keeps_component_count(store: Store):
    for unit in [
        _unit("unit-alpha", "Alpha"),
        _unit("unit-beta", "Beta"),
    ]:
        store.insert_unit(unit)

    result = GraphService(store).analyze_connected_components(limit=1)

    assert result["component_count"] == 2
    assert len(result["components"]) == 1
    assert result["components"][0]["component_id"] == "component-001"


@pytest.mark.parametrize("limit", [-1, True, "many"])
def test_analyze_connected_components_validates_limit(
    store: Store,
    limit: Any,
):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        GraphService(store).analyze_connected_components(limit=limit)


@pytest.mark.parametrize("representative_limit", [-1, True, "many"])
def test_analyze_connected_components_validates_representative_limit(
    store: Store,
    representative_limit: Any,
):
    with pytest.raises(
        ValueError,
        match="representative_limit must be a non-negative integer",
    ):
        GraphService(store).analyze_connected_components(
            representative_limit=representative_limit,
        )
