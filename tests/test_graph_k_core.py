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


def _unit(
    unit_id: str,
    title: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
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
        source=EdgeSource.INFERRED,
    )


def _populate_nested_core_graph(store: Store) -> None:
    for unit in [
        _unit("unit-a", "Alpha"),
        _unit("unit-b", "Beta", source_project=SourceProject.PRESENCE),
        _unit("unit-c", "Gamma"),
        _unit("unit-d", "Delta"),
        _unit("unit-e", "Epsilon"),
        _unit("unit-f", "Phi"),
        _unit("unit-g", "Leaf"),
        _unit("unit-h", "Isolated"),
    ]:
        store.insert_unit(unit)

    for index, (from_unit_id, to_unit_id) in enumerate(
        [
            ("unit-a", "unit-b"),
            ("unit-a", "unit-c"),
            ("unit-a", "unit-d"),
            ("unit-b", "unit-c"),
            ("unit-b", "unit-d"),
            ("unit-c", "unit-d"),
            ("unit-d", "unit-e"),
            ("unit-d", "unit-f"),
            ("unit-e", "unit-f"),
            ("unit-f", "unit-g"),
        ]
    ):
        store.insert_edge(_edge(f"edge-{index}", from_unit_id, to_unit_id))


def test_get_k_core_decomposition_returns_ranked_core_numbers_for_nested_clusters(
    store: Store,
):
    _populate_nested_core_graph(store)

    result = GraphService(store).get_k_core_decomposition()

    assert result["stats"] == {
        "node_count": 8,
        "edge_count": 10,
        "max_core": 3,
        "returned_count": 7,
    }
    assert result["core_groups"] == [
        {
            "core_number": 3,
            "unit_count": 4,
            "unit_ids": ["unit-d", "unit-a", "unit-b", "unit-c"],
        },
        {
            "core_number": 2,
            "unit_count": 2,
            "unit_ids": ["unit-f", "unit-e"],
        },
        {
            "core_number": 1,
            "unit_count": 1,
            "unit_ids": ["unit-g"],
        },
    ]
    assert result["nodes"] == [
        {
            "unit_id": "unit-d",
            "title": "Delta",
            "source_project": "max",
            "core_number": 3,
            "degree": 5,
            "neighbor_count": 5,
        },
        {
            "unit_id": "unit-a",
            "title": "Alpha",
            "source_project": "max",
            "core_number": 3,
            "degree": 3,
            "neighbor_count": 3,
        },
        {
            "unit_id": "unit-b",
            "title": "Beta",
            "source_project": "presence",
            "core_number": 3,
            "degree": 3,
            "neighbor_count": 3,
        },
        {
            "unit_id": "unit-c",
            "title": "Gamma",
            "source_project": "max",
            "core_number": 3,
            "degree": 3,
            "neighbor_count": 3,
        },
        {
            "unit_id": "unit-f",
            "title": "Phi",
            "source_project": "max",
            "core_number": 2,
            "degree": 3,
            "neighbor_count": 3,
        },
        {
            "unit_id": "unit-e",
            "title": "Epsilon",
            "source_project": "max",
            "core_number": 2,
            "degree": 2,
            "neighbor_count": 2,
        },
        {
            "unit_id": "unit-g",
            "title": "Leaf",
            "source_project": "max",
            "core_number": 1,
            "degree": 1,
            "neighbor_count": 1,
        },
    ]


def test_get_k_core_decomposition_filters_by_min_core_and_applies_limit(
    store: Store,
):
    _populate_nested_core_graph(store)

    result = GraphService(store).get_k_core_decomposition(min_core=2, limit=3)

    assert result["stats"] == {
        "node_count": 8,
        "edge_count": 10,
        "max_core": 3,
        "returned_count": 3,
    }
    assert [node["unit_id"] for node in result["nodes"]] == [
        "unit-d",
        "unit-a",
        "unit-b",
    ]
    assert [node["core_number"] for node in result["nodes"]] == [3, 3, 3]


def test_get_k_core_decomposition_empty_graph_returns_zeroed_stats(store: Store):
    result = GraphService(store).get_k_core_decomposition()

    assert result == {
        "stats": {
            "node_count": 0,
            "edge_count": 0,
            "max_core": 0,
            "returned_count": 0,
        },
        "nodes": [],
    }


@pytest.mark.parametrize("min_core", [0, -1, "bad", True])
def test_get_k_core_decomposition_validates_min_core(store: Store, min_core):
    with pytest.raises(ValueError, match="min_core must be a positive integer"):
        GraphService(store).get_k_core_decomposition(min_core=min_core)


@pytest.mark.parametrize("limit", [0, -1, "bad", True])
def test_get_k_core_decomposition_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        GraphService(store).get_k_core_decomposition(limit=limit)
