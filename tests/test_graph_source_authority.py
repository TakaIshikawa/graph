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
    source_project: SourceProject | str = SourceProject.MAX,
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


def test_analyze_source_authority_returns_stats_and_ranked_sources(store: Store):
    for unit in [
        _unit("max-a", "Max Alpha", source_project=SourceProject.MAX),
        _unit("max-b", "Max Beta", source_project=SourceProject.MAX),
        _unit("presence-a", "Presence Alpha", source_project=SourceProject.PRESENCE),
        _unit("forty-two-a", "Forty Two Alpha", source_project=SourceProject.FORTY_TWO),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-1", "max-a", "presence-a"),
        _edge("edge-2", "max-b", "presence-a"),
        _edge("edge-3", "presence-a", "forty-two-a"),
        _edge("edge-4", "forty-two-a", "presence-a"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_source_authority(
        limit=2,
        top_units_per_source=1,
    )

    assert result["stats"] == {
        "source_count": 3,
        "node_count": 4,
        "edge_count": 4,
        "limit": 2,
        "top_units_per_source": 1,
        "pagerank_total": pytest.approx(1.0),
    }
    assert [source["source_project"] for source in result["sources"]] == [
        "presence",
        "forty_two",
    ]

    presence = result["sources"][0]
    assert presence["unit_count"] == 1
    assert presence["incoming_edges"] == 3
    assert presence["outgoing_edges"] == 1
    assert presence["average_degree"] == 4.0
    assert presence["pagerank_sum"] == pytest.approx(
        presence["top_units"][0]["pagerank_score"]
    )
    assert presence["top_units"] == [
        {
            "unit_id": "presence-a",
            "title": "Presence Alpha",
            "pagerank_score": pytest.approx(presence["pagerank_sum"]),
            "incoming_edges": 3,
            "outgoing_edges": 1,
            "degree": 4,
        }
    ]


def test_analyze_source_authority_includes_sources_with_no_edges(store: Store):
    store.insert_unit(_unit("connected", "Connected", source_project=SourceProject.MAX))
    store.insert_unit(_unit("target", "Target", source_project=SourceProject.PRESENCE))
    store.insert_unit(_unit("isolated", "Isolated", source_project="quiet"))
    store.insert_edge(_edge("edge-a", "connected", "target"))

    result = GraphService(store).analyze_source_authority()
    quiet = next(
        source for source in result["sources"] if source["source_project"] == "quiet"
    )

    assert quiet["unit_count"] == 1
    assert quiet["incoming_edges"] == 0
    assert quiet["outgoing_edges"] == 0
    assert quiet["average_degree"] == 0.0
    assert quiet["top_units"][0]["unit_id"] == "isolated"


def test_analyze_source_authority_zero_limits_return_sources_without_units(
    store: Store,
):
    store.insert_unit(_unit("unit-a", "Alpha", source_project=SourceProject.MAX))

    no_sources = GraphService(store).analyze_source_authority(limit=0)
    assert no_sources["sources"] == []
    assert no_sources["stats"]["source_count"] == 1

    no_units = GraphService(store).analyze_source_authority(top_units_per_source=0)
    assert no_units["sources"][0]["top_units"] == []


def test_analyze_source_authority_tie_breaks_by_source_and_unit_id(store: Store):
    for unit in [
        _unit("b-unit", "Same", source_project="b-source"),
        _unit("a-unit", "Same", source_project="a-source"),
    ]:
        store.insert_unit(unit)

    result = GraphService(store).analyze_source_authority()

    assert [source["source_project"] for source in result["sources"]] == [
        "a-source",
        "b-source",
    ]
    assert result["sources"][0]["top_units"][0]["unit_id"] == "a-unit"


@pytest.mark.parametrize("limit", [-1, "bad", None, True])
def test_analyze_source_authority_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_source_authority(limit=limit)


@pytest.mark.parametrize("top_units_per_source", [-1, "bad", None, True])
def test_analyze_source_authority_validates_top_units_per_source(
    store: Store, top_units_per_source
):
    with pytest.raises(
        ValueError,
        match="top_units_per_source must be a non-negative integer",
    ):
        GraphService(store).analyze_source_authority(
            top_units_per_source=top_units_per_source
        )


def test_analyze_source_authority_empty_graph_returns_empty_payload(store: Store):
    assert GraphService(store).analyze_source_authority() == {
        "stats": {
            "source_count": 0,
            "node_count": 0,
            "edge_count": 0,
            "limit": 20,
            "top_units_per_source": 3,
            "pagerank_total": 0.0,
        },
        "sources": [],
    }
