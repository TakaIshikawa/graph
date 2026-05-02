from __future__ import annotations

import os
import tempfile

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _unit(unit_id: str, title: str, tags: list[str] | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        tags=tags or [],
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def test_analyze_tag_bridges_identifies_units_between_tag_communities(store: Store):
    for unit in [
        _unit("unit-bridge", "Bridge"),
        _unit("unit-solar-a", "Solar A", ["solar"]),
        _unit("unit-solar-b", "Solar B", ["solar"]),
        _unit("unit-writing-a", "Writing A", ["writing"]),
        _unit("unit-writing-b", "Writing B", ["writing"]),
        _unit("unit-focus", "Focus"),
        _unit("unit-solar-c", "Solar C", ["solar"]),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-bridge-solar-a", "unit-bridge", "unit-solar-a"),
        _edge("edge-bridge-solar-b", "unit-solar-b", "unit-bridge"),
        _edge("edge-bridge-writing-a", "unit-bridge", "unit-writing-a"),
        _edge("edge-bridge-writing-b", "unit-writing-b", "unit-bridge"),
        _edge("edge-focus-solar-a", "unit-focus", "unit-solar-a"),
        _edge("edge-focus-solar-c", "unit-focus", "unit-solar-c"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_tag_bridges()

    assert result["min_tags"] == 2
    assert result["limit"] is None
    assert result["bridge_count"] == 1
    assert result["bridges"] == [
        {
            "unit_id": "unit-bridge",
            "title": "Bridge",
            "bridge_score": 4,
            "tag_count": 2,
            "neighbor_count": 4,
            "bridging_tags": ["solar", "writing"],
            "evidence_neighbor_ids": [
                "unit-solar-a",
                "unit-solar-b",
                "unit-writing-a",
                "unit-writing-b",
            ],
        }
    ]


def test_analyze_tag_bridges_ignores_sparse_and_untagged_neighbors(store: Store):
    for unit in [
        _unit("unit-center", "Center"),
        _unit("unit-tagged", "Tagged", ["research"]),
        _unit("unit-untagged", "Untagged"),
        _unit("unit-empty-tag", "Empty Tag", [" "]),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-center-tagged", "unit-center", "unit-tagged"),
        _edge("edge-center-untagged", "unit-center", "unit-untagged"),
        _edge("edge-center-empty", "unit-empty-tag", "unit-center"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_tag_bridges()

    assert result["bridge_count"] == 0
    assert result["bridges"] == []


def test_analyze_tag_bridges_applies_limit_and_sorts_equal_scores(store: Store):
    for unit in [
        _unit("unit-beta", "Beta Bridge"),
        _unit("unit-alpha", "Alpha Bridge"),
        _unit("unit-solar", "Solar", ["solar"]),
        _unit("unit-writing", "Writing", ["writing"]),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-beta-solar", "unit-beta", "unit-solar"),
        _edge("edge-beta-writing", "unit-beta", "unit-writing"),
        _edge("edge-alpha-solar", "unit-alpha", "unit-solar"),
        _edge("edge-alpha-writing", "unit-alpha", "unit-writing"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_tag_bridges(limit=1)

    assert result["bridge_count"] == 2
    assert [bridge["unit_id"] for bridge in result["bridges"]] == ["unit-alpha"]


@pytest.mark.parametrize("min_tags", [0, 1, "2", True])
def test_analyze_tag_bridges_validates_min_tags(store: Store, min_tags):
    with pytest.raises(
        ValueError, match="min_tags must be an integer greater than or equal to 2"
    ):
        GraphService(store).analyze_tag_bridges(min_tags=min_tags)


@pytest.mark.parametrize("limit", [-1, "bad", True])
def test_analyze_tag_bridges_validates_limit(store: Store, limit):
    with pytest.raises(
        ValueError, match="limit must be a non-negative integer or None"
    ):
        GraphService(store).analyze_tag_bridges(limit=limit)
