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


def _insert_unit(store: Store, unit_id: str, title: str):
    return store.insert_unit(
        KnowledgeUnit(
            id=unit_id,
            source_project=SourceProject.MAX,
            source_id=unit_id,
            source_entity_type="insight",
            title=title,
            content=f"{title} content",
            content_type=ContentType.INSIGHT,
        )
    )


def _insert_edge(store: Store, from_unit_id: str, to_unit_id: str):
    return store.insert_edge(
        KnowledgeEdge(
            from_unit_id=from_unit_id,
            to_unit_id=to_unit_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.MANUAL,
        )
    )


@pytest.fixture
def eigenvector_store(store: Store):
    units = {
        "center": _insert_unit(store, "center", "Center"),
        "leaf-a": _insert_unit(store, "leaf-a", "Leaf A"),
        "leaf-b": _insert_unit(store, "leaf-b", "Leaf B"),
        "leaf-c": _insert_unit(store, "leaf-c", "Leaf C"),
    }
    for leaf_id in ["leaf-a", "leaf-b", "leaf-c"]:
        _insert_edge(store, units["center"].id, units[leaf_id].id)
        _insert_edge(store, units[leaf_id].id, units["center"].id)
    return store


def test_eigenvector_centrality_ranks_unambiguous_top_node(
    eigenvector_store: Store,
):
    result = GraphService(eigenvector_store).eigenvector_centrality(
        max_iter=1000,
        tolerance=1e-9,
    )

    assert result["stats"] == {
        "node_count": 4,
        "edge_count": 6,
        "max_iter": 1000,
        "tolerance": 1e-9,
        "limit": None,
        "converged": True,
        "error": None,
    }
    assert [node["unit_id"] for node in result["nodes"]] == [
        "center",
        "leaf-a",
        "leaf-b",
        "leaf-c",
    ]
    assert result["nodes"][0]["title"] == "Center"
    assert result["nodes"][0]["score"] > result["nodes"][1]["score"]
    assert result["nodes"][1]["score"] == pytest.approx(result["nodes"][2]["score"])


def test_eigenvector_centrality_applies_optional_limit(eigenvector_store: Store):
    result = GraphService(eigenvector_store).eigenvector_centrality(
        max_iter=1000,
        tolerance=1e-9,
        limit=2,
    )

    assert result["stats"]["limit"] == 2
    assert [node["unit_id"] for node in result["nodes"]] == ["center", "leaf-a"]


def test_eigenvector_centrality_accepts_zero_limit(eigenvector_store: Store):
    result = GraphService(eigenvector_store).eigenvector_centrality(limit=0)

    assert result["stats"]["node_count"] == 4
    assert result["stats"]["limit"] == 0
    assert result["nodes"] == []


def test_eigenvector_centrality_empty_graph_returns_stats(store: Store):
    assert GraphService(store).eigenvector_centrality() == {
        "stats": {
            "node_count": 0,
            "edge_count": 0,
            "max_iter": 100,
            "tolerance": 1e-6,
            "limit": None,
            "converged": True,
            "error": None,
        },
        "nodes": [],
    }


@pytest.mark.parametrize("limit", [-1, "many", True])
def test_eigenvector_centrality_rejects_invalid_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).eigenvector_centrality(limit=limit)


@pytest.mark.parametrize("max_iter", [0, -1, "many", None, True])
def test_eigenvector_centrality_rejects_invalid_max_iter(store: Store, max_iter):
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        GraphService(store).eigenvector_centrality(max_iter=max_iter)


@pytest.mark.parametrize("tolerance", [0, -1.0, "small", None, True])
def test_eigenvector_centrality_rejects_invalid_tolerance(store: Store, tolerance):
    with pytest.raises(ValueError, match="tolerance must be a positive finite number"):
        GraphService(store).eigenvector_centrality(tolerance=tolerance)
