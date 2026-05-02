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


def _unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
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


def test_analyze_reciprocal_links_returns_titles_direction_details_and_weights(
    store: Store,
):
    for unit in [
        _unit("unit-alpha", "Alpha"),
        _unit("unit-beta", "Beta"),
        _unit("unit-gamma", "Gamma"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge(
            "edge-alpha-beta-relates",
            "unit-alpha",
            "unit-beta",
            EdgeRelation.RELATES_TO,
            1.25,
            EdgeSource.MANUAL,
        ),
        _edge(
            "edge-alpha-beta-challenges",
            "unit-alpha",
            "unit-beta",
            EdgeRelation.CHALLENGES,
            0.75,
            EdgeSource.SOURCE,
        ),
        _edge(
            "edge-beta-alpha-builds",
            "unit-beta",
            "unit-alpha",
            EdgeRelation.BUILDS_ON,
            2.0,
            EdgeSource.INFERRED,
        ),
        _edge("edge-alpha-gamma", "unit-alpha", "unit-gamma"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_reciprocal_links()

    assert result["total_pair_count"] == 1
    assert result["returned_count"] == 1
    assert result["limit"] == 20
    pair = result["reciprocal_links"][0]
    assert pair["unit_ids"] == ["unit-alpha", "unit-beta"]
    assert [endpoint["title"] for endpoint in pair["endpoints"]] == ["Alpha", "Beta"]
    assert pair["combined_weight"] == 4.0
    assert pair["edge_count"] == 3
    assert pair["directions"] == [
        {
            "from_unit_id": "unit-alpha",
            "to_unit_id": "unit-beta",
            "from_title": "Alpha",
            "to_title": "Beta",
            "edge_count": 2,
            "total_weight": 2.0,
            "relations": ["challenges", "relates_to"],
            "sources": ["manual", "source"],
            "edges": [
                {
                    "id": "edge-alpha-beta-challenges",
                    "from_unit_id": "unit-alpha",
                    "to_unit_id": "unit-beta",
                    "relation": "challenges",
                    "weight": 0.75,
                    "source": "source",
                    "metadata": {},
                    "created_at": pair["directions"][0]["edges"][0]["created_at"],
                },
                {
                    "id": "edge-alpha-beta-relates",
                    "from_unit_id": "unit-alpha",
                    "to_unit_id": "unit-beta",
                    "relation": "relates_to",
                    "weight": 1.25,
                    "source": "manual",
                    "metadata": {},
                    "created_at": pair["directions"][0]["edges"][1]["created_at"],
                },
            ],
        },
        {
            "from_unit_id": "unit-beta",
            "to_unit_id": "unit-alpha",
            "from_title": "Beta",
            "to_title": "Alpha",
            "edge_count": 1,
            "total_weight": 2.0,
            "relations": ["builds_on"],
            "sources": ["inferred"],
            "edges": [
                {
                    "id": "edge-beta-alpha-builds",
                    "from_unit_id": "unit-beta",
                    "to_unit_id": "unit-alpha",
                    "relation": "builds_on",
                    "weight": 2.0,
                    "source": "inferred",
                    "metadata": {},
                    "created_at": pair["directions"][1]["edges"][0]["created_at"],
                }
            ],
        },
    ]


def test_analyze_reciprocal_links_excludes_one_way_edges_and_self_links(
    store: Store,
):
    for unit in [
        _unit("unit-alpha", "Alpha"),
        _unit("unit-beta", "Beta"),
        _unit("unit-gamma", "Gamma"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-alpha-beta", "unit-alpha", "unit-beta"),
        _edge("edge-gamma-gamma", "unit-gamma", "unit-gamma"),
    ]:
        store.insert_edge(edge)

    assert GraphService(store).analyze_reciprocal_links() == {
        "total_pair_count": 0,
        "returned_count": 0,
        "limit": 20,
        "reciprocal_links": [],
    }


def test_analyze_reciprocal_links_orders_by_combined_weight_then_unit_ids(
    store: Store,
):
    for unit in [
        _unit("unit-a", "A"),
        _unit("unit-b", "B"),
        _unit("unit-c", "C"),
        _unit("unit-d", "D"),
        _unit("unit-e", "E"),
        _unit("unit-f", "F"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-c-d", "unit-c", "unit-d", weight=2.0),
        _edge("edge-d-c", "unit-d", "unit-c", weight=2.0),
        _edge("edge-a-b", "unit-a", "unit-b", weight=1.5),
        _edge("edge-b-a", "unit-b", "unit-a", weight=1.5),
        _edge("edge-e-f", "unit-e", "unit-f", weight=1.0),
        _edge("edge-f-e", "unit-f", "unit-e", weight=2.0),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_reciprocal_links()

    assert [pair["unit_ids"] for pair in result["reciprocal_links"]] == [
        ["unit-c", "unit-d"],
        ["unit-a", "unit-b"],
        ["unit-e", "unit-f"],
    ]
    assert [pair["combined_weight"] for pair in result["reciprocal_links"]] == [
        4.0,
        3.0,
        3.0,
    ]


def test_analyze_reciprocal_links_applies_limit_after_total_count(store: Store):
    for unit in [
        _unit("unit-a", "A"),
        _unit("unit-b", "B"),
        _unit("unit-c", "C"),
        _unit("unit-d", "D"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-a-b", "unit-a", "unit-b", weight=3.0),
        _edge("edge-b-a", "unit-b", "unit-a", weight=3.0),
        _edge("edge-c-d", "unit-c", "unit-d", weight=1.0),
        _edge("edge-d-c", "unit-d", "unit-c", weight=1.0),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_reciprocal_links(limit=1)

    assert result["total_pair_count"] == 2
    assert result["returned_count"] == 1
    assert result["limit"] == 1
    assert [pair["unit_ids"] for pair in result["reciprocal_links"]] == [
        ["unit-a", "unit-b"]
    ]


def test_analyze_reciprocal_links_accepts_zero_limit(store: Store):
    for unit in [_unit("unit-a", "A"), _unit("unit-b", "B")]:
        store.insert_unit(unit)
    store.insert_edge(_edge("edge-a-b", "unit-a", "unit-b"))
    store.insert_edge(_edge("edge-b-a", "unit-b", "unit-a"))

    assert GraphService(store).analyze_reciprocal_links(limit=0) == {
        "total_pair_count": 1,
        "returned_count": 0,
        "limit": 0,
        "reciprocal_links": [],
    }


@pytest.mark.parametrize("limit", [-1, 1.5, "1", None, True])
def test_analyze_reciprocal_links_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_reciprocal_links(limit=limit)
