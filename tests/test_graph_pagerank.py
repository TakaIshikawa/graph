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


def _insert_edge(
    store: Store,
    from_unit_id: str,
    to_unit_id: str,
    *,
    relation: EdgeRelation = EdgeRelation.REFERENCES,
    weight: float = 1.0,
):
    return store.insert_edge(
        KnowledgeEdge(
            from_unit_id=from_unit_id,
            to_unit_id=to_unit_id,
            relation=relation,
            weight=weight,
            source=EdgeSource.MANUAL,
        )
    )


def test_pagerank_centrality_returns_ranked_unit_metadata(store: Store):
    source = _insert_unit(store, "source", "Source")
    target = _insert_unit(store, "target", "Target")
    isolated = _insert_unit(store, "isolated", "Isolated")
    _insert_edge(store, source.id, target.id)

    results = GraphService(store).pagerank_centrality(top_n=1)

    assert [item["id"] for item in results] == ["target"]
    assert results[0] == {
        "id": "target",
        "source_project": "max",
        "source_id": "target",
        "source_entity_type": "insight",
        "title": "Target",
        "content_type": "insight",
        "score": results[0]["score"],
        "in_degree": 1,
        "out_degree": 0,
    }
    assert isolated.id not in {item["id"] for item in results}


def test_pagerank_centrality_tie_breaks_by_unit_id(store: Store):
    _insert_unit(store, "unit-b", "Beta")
    _insert_unit(store, "unit-a", "Alpha")

    results = GraphService(store).pagerank_centrality()

    assert [item["id"] for item in results] == ["unit-a", "unit-b"]
    assert results[0]["score"] == pytest.approx(results[1]["score"])
    assert results[0]["in_degree"] == results[0]["out_degree"] == 0


def test_pagerank_centrality_relation_filter_limits_edges_and_degrees(store: Store):
    hub = _insert_unit(store, "hub", "Hub")
    referenced = _insert_unit(store, "referenced", "Referenced")
    inspired = _insert_unit(store, "inspired", "Inspired")
    _insert_edge(store, hub.id, referenced.id, relation=EdgeRelation.REFERENCES)
    _insert_edge(store, hub.id, inspired.id, relation=EdgeRelation.INSPIRES)

    results = GraphService(store).pagerank_centrality(
        relation_filter=EdgeRelation.REFERENCES
    )
    by_id = {item["id"]: item for item in results}

    assert by_id["referenced"]["score"] > by_id["inspired"]["score"]
    assert by_id["referenced"]["in_degree"] == 1
    assert by_id["inspired"]["in_degree"] == 0
    assert by_id["hub"]["out_degree"] == 1


def test_pagerank_centrality_empty_graph_returns_empty_result(store: Store):
    assert GraphService(store).pagerank_centrality() == []


def test_pagerank_centrality_weighted_and_unweighted_modes(store: Store):
    source = _insert_unit(store, "source", "Source")
    heavy = _insert_unit(store, "heavy", "Heavy")
    light = _insert_unit(store, "light", "Light")
    _insert_edge(store, source.id, heavy.id, weight=10.0)
    _insert_edge(store, source.id, light.id, weight=1.0)

    weighted = {
        item["id"]: item["score"]
        for item in GraphService(store).pagerank_centrality(weight="weight")
    }
    unweighted = {
        item["id"]: item["score"]
        for item in GraphService(store).pagerank_centrality(weight=None)
    }

    assert weighted["heavy"] > weighted["light"]
    assert unweighted["heavy"] == pytest.approx(unweighted["light"])
