"""Tests for triangle motif analysis."""

from __future__ import annotations

import os
import tempfile

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _unit(unit_id: str, title: str, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def _edge(
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
    weight: float,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=weight,
    )


@pytest.fixture
def triangle_store(store: Store):
    for unit in [
        _unit("unit-alpha", "Alpha storage", ["energy", "solar", "storage"]),
        _unit("unit-beta", "Beta grid", ["energy", "solar", "grid"]),
        _unit("unit-gamma", "Gamma battery", ["energy", "solar", "storage"]),
        _unit("unit-delta", "Delta draft", ["writing", "draft"]),
        _unit("unit-epsilon", "Epsilon review", ["writing", "review"]),
        _unit("unit-zeta", "Zeta edit", ["writing", "draft", "review"]),
        _unit("unit-open-a", "Open A", ["energy", "open"]),
        _unit("unit-open-b", "Open B", ["energy", "open"]),
        _unit("unit-open-c", "Open C", ["energy", "open"]),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("unit-alpha", "unit-beta", EdgeRelation.BUILDS_ON, 0.9),
        _edge("unit-beta", "unit-gamma", EdgeRelation.INSPIRES, 0.8),
        _edge("unit-gamma", "unit-alpha", EdgeRelation.REFERENCES, 0.7),
        _edge("unit-delta", "unit-epsilon", EdgeRelation.RELATES_TO, 0.6),
        _edge("unit-epsilon", "unit-zeta", EdgeRelation.REFINES, 0.6),
        _edge("unit-delta", "unit-zeta", EdgeRelation.CHALLENGES, 0.6),
        _edge("unit-open-a", "unit-open-b", EdgeRelation.RELATES_TO, 1.0),
        _edge("unit-open-b", "unit-open-c", EdgeRelation.RELATES_TO, 1.0),
    ]:
        store.insert_edge(edge)

    return store


def test_analyze_triangles_returns_closed_motifs_with_details(triangle_store: Store):
    result = GraphService(triangle_store).analyze_triangles()

    assert result == [
        {
            "unit_ids": ["unit-alpha", "unit-beta", "unit-gamma"],
            "titles": ["Alpha storage", "Beta grid", "Gamma battery"],
            "shared_tags": ["energy", "solar"],
            "relations": [
                {
                    "unit_ids": ["unit-alpha", "unit-beta"],
                    "labels": ["builds_on"],
                    "weight": 0.9,
                },
                {
                    "unit_ids": ["unit-alpha", "unit-gamma"],
                    "labels": ["references"],
                    "weight": 0.7,
                },
                {
                    "unit_ids": ["unit-beta", "unit-gamma"],
                    "labels": ["inspires"],
                    "weight": 0.8,
                },
            ],
            "score": 1.3,
        },
        {
            "unit_ids": ["unit-delta", "unit-epsilon", "unit-zeta"],
            "titles": ["Delta draft", "Epsilon review", "Zeta edit"],
            "shared_tags": ["writing"],
            "relations": [
                {
                    "unit_ids": ["unit-delta", "unit-epsilon"],
                    "labels": ["relates_to"],
                    "weight": 0.6,
                },
                {
                    "unit_ids": ["unit-delta", "unit-zeta"],
                    "labels": ["challenges"],
                    "weight": 0.6,
                },
                {
                    "unit_ids": ["unit-epsilon", "unit-zeta"],
                    "labels": ["refines"],
                    "weight": 0.6,
                },
            ],
            "score": 0.933333,
        },
    ]


def test_analyze_triangles_excludes_open_triples(triangle_store: Store):
    result = GraphService(triangle_store).analyze_triangles()

    assert ["unit-open-a", "unit-open-b", "unit-open-c"] not in [
        motif["unit_ids"] for motif in result
    ]


def test_analyze_triangles_filters_by_tag_and_min_weight(triangle_store: Store):
    service = GraphService(triangle_store)

    writing = service.analyze_triangles(tag="writing")
    heavy = service.analyze_triangles(min_weight=0.65)

    assert [motif["unit_ids"] for motif in writing] == [
        ["unit-delta", "unit-epsilon", "unit-zeta"]
    ]
    assert [motif["unit_ids"] for motif in heavy] == [
        ["unit-alpha", "unit-beta", "unit-gamma"]
    ]


def test_analyze_triangles_is_deterministic_and_limited(triangle_store: Store):
    service = GraphService(triangle_store)

    first = service.analyze_triangles(limit=1)
    second = GraphService(triangle_store).analyze_triangles(limit=1)

    assert first == second
    assert [motif["unit_ids"] for motif in first] == [
        ["unit-alpha", "unit-beta", "unit-gamma"]
    ]

