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


def _insert_unit(
    store: Store,
    unit_id: str,
    title: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
):
    return store.insert_unit(
        KnowledgeUnit(
            id=unit_id,
            source_project=source_project,
            source_id=unit_id,
            source_entity_type="insight",
            title=title,
            content=f"{title} content",
            content_type=ContentType.INSIGHT,
        )
    )


def _insert_edge(store: Store, from_unit_id: str, to_unit_id: str, weight: float = 1.0):
    return store.insert_edge(
        KnowledgeEdge(
            from_unit_id=from_unit_id,
            to_unit_id=to_unit_id,
            relation=EdgeRelation.RELATES_TO,
            weight=weight,
            source=EdgeSource.MANUAL,
        )
    )


@pytest.fixture
def betweenness_store(store: Store):
    units = {
        "leaf-a": _insert_unit(store, "leaf-a", "Leaf A"),
        "bridge-a": _insert_unit(
            store,
            "bridge-a",
            "Bridge A",
            source_project=SourceProject.FORTY_TWO,
        ),
        "bridge-b": _insert_unit(
            store,
            "bridge-b",
            "Bridge B",
            source_project=SourceProject.PRESENCE,
        ),
        "leaf-b": _insert_unit(store, "leaf-b", "Leaf B"),
        "isolated": _insert_unit(store, "isolated", "Isolated"),
    }
    for left_id, right_id in [
        ("leaf-a", "bridge-a"),
        ("bridge-a", "bridge-b"),
        ("bridge-b", "leaf-b"),
    ]:
        _insert_edge(store, units[left_id].id, units[right_id].id)
    return store


def test_betweenness_centrality_ranks_bridge_units_above_leaves(
    betweenness_store: Store,
):
    result = GraphService(betweenness_store).betweenness_centrality(limit=None)

    assert result["stats"] == {
        "node_count": 5,
        "edge_count": 3,
        "limit": None,
        "normalized": True,
        "weight": None,
    }
    assert [node["unit_id"] for node in result["nodes"]] == [
        "bridge-a",
        "bridge-b",
        "isolated",
        "leaf-a",
        "leaf-b",
    ]
    assert result["nodes"][0] == {
        "unit_id": "bridge-a",
        "title": "Bridge A",
        "score": pytest.approx(0.3333333333333333),
        "source_project": "forty_two",
        "degree": 2,
        "in_degree": 1,
        "out_degree": 1,
    }
    assert result["nodes"][0]["score"] == pytest.approx(result["nodes"][1]["score"])
    assert result["nodes"][0]["score"] > result["nodes"][3]["score"]


def test_betweenness_centrality_applies_limit_with_deterministic_ties(
    betweenness_store: Store,
):
    result = GraphService(betweenness_store).betweenness_centrality(limit=2)

    assert result["stats"]["limit"] == 2
    assert [node["unit_id"] for node in result["nodes"]] == [
        "bridge-a",
        "bridge-b",
    ]


def test_betweenness_centrality_accepts_zero_limit(betweenness_store: Store):
    result = GraphService(betweenness_store).betweenness_centrality(limit=0)

    assert result["stats"]["node_count"] == 5
    assert result["stats"]["limit"] == 0
    assert result["nodes"] == []


def test_betweenness_centrality_empty_graph_returns_stats(store: Store):
    assert GraphService(store).betweenness_centrality() == {
        "stats": {
            "node_count": 0,
            "edge_count": 0,
            "limit": 10,
            "normalized": True,
            "weight": None,
        },
        "nodes": [],
    }


def test_betweenness_centrality_handles_isolated_units(store: Store):
    _insert_unit(store, "isolated-a", "Isolated A")
    _insert_unit(store, "isolated-b", "Isolated B")

    result = GraphService(store).betweenness_centrality(limit=None)

    assert [node["unit_id"] for node in result["nodes"]] == [
        "isolated-a",
        "isolated-b",
    ]
    assert [node["score"] for node in result["nodes"]] == [0.0, 0.0]


@pytest.mark.parametrize("limit", [-1, "many", True])
def test_betweenness_centrality_rejects_invalid_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        GraphService(store).betweenness_centrality(limit=limit)


@pytest.mark.parametrize("normalized", [0, 1, "yes", None])
def test_betweenness_centrality_rejects_invalid_normalized(store: Store, normalized):
    with pytest.raises(ValueError, match="normalized must be a boolean"):
        GraphService(store).betweenness_centrality(normalized=normalized)


@pytest.mark.parametrize("weight", [1, 1.5, object()])
def test_betweenness_centrality_rejects_invalid_weight(store: Store, weight):
    with pytest.raises(ValueError, match="weight must be a string, True, False, or None"):
        GraphService(store).betweenness_centrality(weight=weight)
