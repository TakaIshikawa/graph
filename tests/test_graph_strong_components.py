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
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.INFERRED,
    )


def _populate(store: Store) -> None:
    for unit in [
        _unit("unit-a", "Alpha", tags=["loop", "alpha"]),
        _unit("unit-b", "Beta", tags=["loop", "beta"]),
        _unit(
            "unit-c",
            "Gamma",
            source_project=SourceProject.PRESENCE,
            tags=["loop", "alpha"],
        ),
        _unit("unit-d", "Delta", tags=["side", "delta"]),
        _unit("unit-e", "Epsilon", tags=["side", "epsilon"]),
        _unit("unit-f", "Forward", tags=["open"]),
        _unit("unit-g", "Sink", tags=["open"]),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-a-b", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
        _edge("edge-b-c", "unit-b", "unit-c", EdgeRelation.INSPIRES),
        _edge("edge-c-a", "unit-c", "unit-a", EdgeRelation.REFERENCES),
        _edge("edge-c-d", "unit-c", "unit-d", EdgeRelation.RELATES_TO),
        _edge("edge-d-e", "unit-d", "unit-e", EdgeRelation.CHALLENGES),
        _edge("edge-e-d", "unit-e", "unit-d", EdgeRelation.REFINES),
        _edge("edge-f-g", "unit-f", "unit-g", EdgeRelation.DISCOVERS),
    ]:
        store.insert_edge(edge)


def test_analyze_strongly_connected_components_returns_directed_component_summaries(
    store: Store,
):
    _populate(store)

    result = GraphService(store).analyze_strongly_connected_components()

    assert result == [
        {
            "size": 3,
            "unit_ids": ["unit-a", "unit-b", "unit-c"],
            "titles": ["Alpha", "Beta", "Gamma"],
            "source_project_counts": {"max": 2, "presence": 1},
            "relation_counts": {
                "builds_on": 1,
                "inspires": 1,
                "references": 1,
            },
            "representative_tags": ["loop", "alpha", "beta"],
        },
        {
            "size": 2,
            "unit_ids": ["unit-d", "unit-e"],
            "titles": ["Delta", "Epsilon"],
            "source_project_counts": {"max": 2},
            "relation_counts": {"challenges": 1, "refines": 1},
            "representative_tags": ["side", "delta", "epsilon"],
        },
    ]


def test_analyze_strongly_connected_components_filters_by_min_size(store: Store):
    _populate(store)

    result = GraphService(store).analyze_strongly_connected_components(min_size=3)

    assert [component["unit_ids"] for component in result] == [
        ["unit-a", "unit-b", "unit-c"]
    ]


def test_analyze_strongly_connected_components_sorts_ties_and_applies_limit(
    store: Store,
):
    for unit_id in ["unit-a", "unit-b", "unit-c", "unit-d"]:
        store.insert_unit(_unit(unit_id, unit_id.title()))
    for edge in [
        _edge("edge-a-b", "unit-a", "unit-b", EdgeRelation.RELATES_TO),
        _edge("edge-b-a", "unit-b", "unit-a", EdgeRelation.RELATES_TO),
        _edge("edge-c-d", "unit-c", "unit-d", EdgeRelation.RELATES_TO),
        _edge("edge-d-c", "unit-d", "unit-c", EdgeRelation.RELATES_TO),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_strongly_connected_components(limit=1)

    assert [component["unit_ids"] for component in result] == [["unit-a", "unit-b"]]


def test_analyze_strongly_connected_components_accepts_zero_limit(store: Store):
    _populate(store)

    assert GraphService(store).analyze_strongly_connected_components(limit=0) == []


@pytest.mark.parametrize("min_size", [0, -1, "bad", True])
def test_analyze_strongly_connected_components_validates_min_size(
    store: Store,
    min_size,
):
    with pytest.raises(ValueError, match="min_size must be a positive integer"):
        GraphService(store).analyze_strongly_connected_components(min_size=min_size)


@pytest.mark.parametrize("limit", [-1, "bad", True])
def test_analyze_strongly_connected_components_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_strongly_connected_components(limit=limit)
