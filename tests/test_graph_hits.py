from __future__ import annotations

import os
import tempfile

import networkx as nx
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
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.MANUAL,
        )
    )


@pytest.fixture
def hits_store(store: Store):
    units = {
        "hub-a": _insert_unit(store, "hub-a", "Primary Hub"),
        "hub-b": _insert_unit(store, "hub-b", "Secondary Hub B"),
        "hub-c": _insert_unit(store, "hub-c", "Secondary Hub C"),
        "leaf": _insert_unit(store, "leaf", "Leaf Source"),
        "authority-a": _insert_unit(store, "authority-a", "Primary Authority"),
        "authority-b": _insert_unit(store, "authority-b", "Secondary Authority"),
        "authority-c": _insert_unit(store, "authority-c", "Tertiary Authority"),
    }

    for from_id, to_id in [
        ("hub-a", "authority-a"),
        ("hub-a", "authority-b"),
        ("hub-a", "authority-c"),
        ("hub-b", "authority-a"),
        ("hub-c", "authority-a"),
        ("leaf", "authority-b"),
    ]:
        _insert_edge(store, units[from_id].id, units[to_id].id)

    return store


def test_analyze_hits_returns_authority_and_hub_rankings(hits_store: Store):
    result = GraphService(hits_store).analyze_hits(limit=3)

    assert result["stats"] == {
        "node_count": 7,
        "edge_count": 6,
        "normalized": True,
        "converged": True,
        "max_iter": 100,
        "error": None,
    }
    assert [unit["unit_id"] for unit in result["authorities"]] == [
        "authority-a",
        "authority-b",
        "authority-c",
    ]
    assert [unit["unit_id"] for unit in result["hubs"]] == [
        "hub-a",
        "hub-b",
        "hub-c",
    ]
    assert result["authorities"][0]["title"] == "Primary Authority"
    assert result["authorities"][0]["authority_score"] > result["authorities"][1][
        "authority_score"
    ]
    assert result["hubs"][0]["title"] == "Primary Hub"
    assert result["hubs"][0]["hub_score"] > result["hubs"][1]["hub_score"]


def test_analyze_hits_tie_breaks_by_unit_id(store: Store):
    hub_b = _insert_unit(store, "hub-b", "Hub B")
    hub_a = _insert_unit(store, "hub-a", "Hub A")
    authority = _insert_unit(store, "authority", "Authority")
    _insert_edge(store, hub_b.id, authority.id)
    _insert_edge(store, hub_a.id, authority.id)

    result = GraphService(store).analyze_hits(limit=2)

    assert [unit["unit_id"] for unit in result["hubs"]] == ["hub-a", "hub-b"]
    assert result["hubs"][0]["hub_score"] == pytest.approx(
        result["hubs"][1]["hub_score"]
    )


def test_analyze_hits_accepts_zero_limit(hits_store: Store):
    result = GraphService(hits_store).analyze_hits(limit=0)

    assert result["stats"]["node_count"] == 7
    assert result["stats"]["edge_count"] == 6
    assert result["authorities"] == []
    assert result["hubs"] == []


@pytest.mark.parametrize("limit", [-1, "many", None, True])
def test_analyze_hits_rejects_invalid_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_hits(limit=limit)


def test_analyze_hits_empty_graph_returns_empty_rankings(store: Store):
    assert GraphService(store).analyze_hits() == {
        "stats": {
            "node_count": 0,
            "edge_count": 0,
            "normalized": True,
            "converged": True,
            "max_iter": 0,
            "error": None,
        },
        "authorities": [],
        "hubs": [],
    }


def test_analyze_hits_retries_when_networkx_does_not_converge(
    store: Store, monkeypatch: pytest.MonkeyPatch
):
    hub = _insert_unit(store, "hub", "Hub")
    authority = _insert_unit(store, "authority", "Authority")
    _insert_edge(store, hub.id, authority.id)
    calls: list[int] = []

    def fake_hits(graph, *, max_iter, normalized):
        calls.append(max_iter)
        if max_iter == 100:
            raise nx.PowerIterationFailedConvergence(max_iter)
        return (
            {"hub": 1.0, "authority": 0.0},
            {"hub": 0.0, "authority": 1.0},
        )

    monkeypatch.setattr(nx, "hits", fake_hits)

    result = GraphService(store).analyze_hits()

    assert calls == [100, 1000]
    assert result["stats"]["converged"] is True
    assert result["stats"]["max_iter"] == 1000
    assert [unit["unit_id"] for unit in result["authorities"][:2]] == [
        "authority",
        "hub",
    ]

