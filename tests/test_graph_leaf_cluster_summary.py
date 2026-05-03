from __future__ import annotations

import os
import tempfile

import pytest

from graph.graph.service import GraphService, analyze_leaf_cluster_summary
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


def test_analyze_leaf_cluster_summary_handles_empty_graph():
    assert analyze_leaf_cluster_summary([], []) == {
        "node_count": 0,
        "edge_count": 0,
        "total_leaf_count": 0,
        "isolated_leaf_count": 0,
        "attached_leaf_count": 0,
        "parent_count": 0,
        "component_leaf_count": 0,
        "parents": [],
        "isolated_leaves": [],
        "leaf_components": [],
        "filters": {"parent_limit": 20, "sample_leaf_limit": 5},
    }


def test_leaf_cluster_summary_distinguishes_isolated_and_attached_leaves(
    store: Store,
):
    for unit in [
        _unit("unit-hub", "Hub"),
        _unit("unit-leaf-a", "Leaf A"),
        _unit("unit-leaf-b", "Leaf B"),
        _unit("unit-middle", "Middle"),
        _unit("unit-isolated", "Isolated"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-a", "unit-leaf-a", "unit-hub"),
        _edge("edge-b", "unit-hub", "unit-leaf-b"),
        _edge("edge-middle", "unit-hub", "unit-middle"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).leaf_cluster_summary(sample_leaf_limit=1)

    assert result["node_count"] == 5
    assert result["edge_count"] == 3
    assert result["total_leaf_count"] == 4
    assert result["isolated_leaf_count"] == 1
    assert result["attached_leaf_count"] == 3
    assert result["parent_count"] == 1
    assert result["component_leaf_count"] == 0
    assert result["isolated_leaves"] == [
        {"unit_id": "unit-isolated", "title": "Isolated", "degree": 0}
    ]
    assert result["parents"] == [
        {
            "unit_id": "unit-hub",
            "title": "Hub",
            "leaf_count": 3,
            "sample_leaves": [
                {"unit_id": "unit-leaf-a", "title": "Leaf A", "degree": 1}
            ],
        }
    ]


def test_leaf_cluster_summary_treats_directed_edges_as_undirected(store: Store):
    for unit in [
        _unit("unit-parent", "Parent"),
        _unit("unit-leaf-in", "Incoming Leaf"),
        _unit("unit-leaf-out", "Outgoing Leaf"),
        _unit("unit-bridge", "Bridge"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-in", "unit-leaf-in", "unit-parent"),
        _edge("edge-out", "unit-parent", "unit-leaf-out"),
        _edge("edge-bridge", "unit-bridge", "unit-parent"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).leaf_cluster_summary()

    assert result["parents"][0]["unit_id"] == "unit-parent"
    assert result["parents"][0]["leaf_count"] == 3
    assert [leaf["unit_id"] for leaf in result["parents"][0]["sample_leaves"]] == [
        "unit-bridge",
        "unit-leaf-in",
        "unit-leaf-out",
    ]


def test_leaf_cluster_summary_groups_two_node_leaf_components(store: Store):
    for unit in [
        _unit("unit-alpha", "Alpha"),
        _unit("unit-beta", "Beta"),
    ]:
        store.insert_unit(unit)
    store.insert_edge(_edge("edge-alpha-beta", "unit-alpha", "unit-beta"))

    result = GraphService(store).leaf_cluster_summary()

    assert result["total_leaf_count"] == 2
    assert result["parent_count"] == 0
    assert result["component_leaf_count"] == 2
    assert result["leaf_components"] == [
        {
            "component_id": "leaf-component-001",
            "leaf_count": 2,
            "sample_leaves": [
                {"unit_id": "unit-alpha", "title": "Alpha", "degree": 1},
                {"unit_id": "unit-beta", "title": "Beta", "degree": 1},
            ],
        }
    ]


def test_leaf_cluster_summary_orders_parent_ties_deterministically(store: Store):
    for unit in [
        _unit("unit-z-parent", "Parent"),
        _unit("unit-a-parent", "Parent"),
        _unit("unit-z-leaf", "Z Leaf"),
        _unit("unit-a-leaf", "A Leaf"),
        _unit("unit-z-core", "Z Core"),
        _unit("unit-a-core", "A Core"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-z-leaf", "unit-z-parent", "unit-z-leaf"),
        _edge("edge-z-core", "unit-z-parent", "unit-z-core"),
        _edge("edge-a-leaf", "unit-a-parent", "unit-a-leaf"),
        _edge("edge-a-core", "unit-a-parent", "unit-a-core"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).leaf_cluster_summary(parent_limit=1)

    assert result["parent_count"] == 2
    assert [parent["unit_id"] for parent in result["parents"]] == ["unit-a-parent"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"parent_limit": -1}, "parent_limit"),
        ({"parent_limit": True}, "parent_limit"),
        ({"parent_limit": "many"}, "parent_limit"),
        ({"sample_leaf_limit": -1}, "sample_leaf_limit"),
        ({"sample_leaf_limit": False}, "sample_leaf_limit"),
    ],
)
def test_leaf_cluster_summary_validates_limits(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        analyze_leaf_cluster_summary([], [], **kwargs)
