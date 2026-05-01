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


def _populate_broker_graph(store: Store) -> None:
    for unit in [
        _unit("unit-a", "Alpha", source_project=SourceProject.MAX),
        _unit("unit-b", "Beta", source_project=SourceProject.PRESENCE),
        _unit("unit-c", "Connector", source_project=SourceProject.FORTY_TWO),
        _unit("unit-d", "Broker", source_project=SourceProject.MAX),
        _unit("unit-e", "Epsilon", source_project=SourceProject.PRESENCE),
        _unit("unit-f", "Phi", source_project=SourceProject.MAX),
        _unit("unit-g", "Leaf", source_project=SourceProject.CSV),
        _unit("unit-h", "Isolated", source_project=SourceProject.JSONL),
    ]:
        store.insert_unit(unit)

    for index, (from_unit_id, to_unit_id) in enumerate(
        [
            ("unit-a", "unit-c"),
            ("unit-b", "unit-c"),
            ("unit-c", "unit-d"),
            ("unit-d", "unit-e"),
            ("unit-d", "unit-f"),
            ("unit-e", "unit-f"),
            ("unit-d", "unit-g"),
        ]
    ):
        store.insert_edge(_edge(f"edge-{index}", from_unit_id, to_unit_id))


def test_analyze_brokers_returns_ranked_betweenness_candidates(store: Store):
    _populate_broker_graph(store)

    result = GraphService(store).analyze_brokers()

    assert result == [
        {
            "unit_id": "unit-d",
            "title": "Broker",
            "source_project": "max",
            "score": 0.52381,
            "degree": 4,
            "neighbor_source_project_diversity": 4,
            "explanation": "Connects 4 neighboring units across 4 source projects.",
        },
        {
            "unit_id": "unit-c",
            "title": "Connector",
            "source_project": "forty_two",
            "score": 0.428571,
            "degree": 3,
            "neighbor_source_project_diversity": 2,
            "explanation": "Connects 3 neighboring units across 2 source projects.",
        },
    ]


def test_analyze_brokers_orders_score_ties_by_unit_id_in_disconnected_graph(
    store: Store,
):
    for unit in [
        _unit("unit-a", "Alpha"),
        _unit("unit-b", "Beta"),
        _unit("unit-c", "Gamma"),
        _unit("unit-x", "Xray"),
        _unit("unit-y", "Yankee"),
        _unit("unit-z", "Zulu"),
        _unit("unit-isolated", "Isolated"),
    ]:
        store.insert_unit(unit)

    for index, (from_unit_id, to_unit_id) in enumerate(
        [
            ("unit-a", "unit-b"),
            ("unit-b", "unit-c"),
            ("unit-x", "unit-y"),
            ("unit-y", "unit-z"),
        ]
    ):
        store.insert_edge(_edge(f"edge-{index}", from_unit_id, to_unit_id))

    result = GraphService(store).analyze_brokers()

    assert [candidate["unit_id"] for candidate in result] == ["unit-b", "unit-y"]
    assert [candidate["score"] for candidate in result] == [0.066667, 0.066667]


def test_analyze_brokers_applies_limit(store: Store):
    _populate_broker_graph(store)

    result = GraphService(store).analyze_brokers(limit=1)

    assert [candidate["unit_id"] for candidate in result] == ["unit-d"]


@pytest.mark.parametrize("limit", [0, "0"])
def test_analyze_brokers_accepts_zero_limit(store: Store, limit):
    _populate_broker_graph(store)

    assert GraphService(store).analyze_brokers(limit=limit) == []


@pytest.mark.parametrize("limit", [-1, "bad", None, True])
def test_analyze_brokers_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_brokers(limit=limit)


def test_analyze_brokers_empty_graph_returns_empty_list(store: Store):
    assert GraphService(store).analyze_brokers() == []


def test_analyze_brokers_returns_empty_when_no_bridge_candidates(store: Store):
    for unit in [
        _unit("unit-a", "Alpha"),
        _unit("unit-b", "Beta"),
        _unit("unit-c", "Gamma"),
        _unit("unit-isolated", "Isolated"),
    ]:
        store.insert_unit(unit)

    for index, (from_unit_id, to_unit_id) in enumerate(
        [
            ("unit-a", "unit-b"),
            ("unit-b", "unit-c"),
            ("unit-c", "unit-a"),
        ]
    ):
        store.insert_edge(_edge(f"edge-{index}", from_unit_id, to_unit_id))

    assert GraphService(store).analyze_brokers() == []
