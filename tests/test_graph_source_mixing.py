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


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.INFERRED,
    )


def test_analyze_source_mixing_returns_aggregate_and_pair_metrics(store: Store):
    for unit in [
        _unit("unit-a", "Alpha", source_project=SourceProject.MAX),
        _unit("unit-b", "Beta", source_project=SourceProject.MAX),
        _unit("unit-c", "Gamma", source_project=SourceProject.PRESENCE),
        _unit("unit-d", "Delta", source_project=SourceProject.FORTY_TWO),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-004", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
        _edge("edge-002", "unit-a", "unit-c", EdgeRelation.INSPIRES),
        _edge("edge-001", "unit-b", "unit-c", EdgeRelation.INSPIRES),
        _edge("edge-003", "unit-c", "unit-a", EdgeRelation.REFERENCES),
        _edge("edge-005", "unit-c", "unit-d", EdgeRelation.RELATES_TO),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_source_mixing(limit=1)

    assert result["total_edge_count"] == 5
    assert result["same_source_edge_count"] == 1
    assert result["cross_source_edge_count"] == 4
    assert result["cross_source_ratio"] == 0.8
    assert result["source_pairs"] == [
        {
            "from_source": "max",
            "to_source": "presence",
            "edge_count": 2,
            "relation_counts": {"inspires": 2},
            "example_edge_ids": ["edge-001"],
        },
        {
            "from_source": "max",
            "to_source": "max",
            "edge_count": 1,
            "relation_counts": {"builds_on": 1},
            "example_edge_ids": ["edge-004"],
        },
        {
            "from_source": "presence",
            "to_source": "forty_two",
            "edge_count": 1,
            "relation_counts": {"relates_to": 1},
            "example_edge_ids": ["edge-005"],
        },
        {
            "from_source": "presence",
            "to_source": "max",
            "edge_count": 1,
            "relation_counts": {"references": 1},
            "example_edge_ids": ["edge-003"],
        },
    ]


def test_analyze_source_mixing_sorts_ties_by_source_names(store: Store):
    for unit in [
        _unit("unit-a", "Alpha", source_project=SourceProject.MAX),
        _unit("unit-b", "Beta", source_project=SourceProject.PRESENCE),
        _unit("unit-c", "Gamma", source_project=SourceProject.FORTY_TWO),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-b", "unit-b", "unit-a"),
        _edge("edge-a", "unit-c", "unit-a"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_source_mixing()

    assert [
        (pair["from_source"], pair["to_source"], pair["edge_count"])
        for pair in result["source_pairs"]
    ] == [
        ("forty_two", "max", 1),
        ("presence", "max", 1),
    ]


def test_analyze_source_mixing_zero_limit_returns_pairs_without_examples(store: Store):
    store.insert_unit(_unit("unit-a", "Alpha"))
    store.insert_unit(_unit("unit-b", "Beta", source_project=SourceProject.PRESENCE))
    store.insert_edge(_edge("edge-a", "unit-a", "unit-b"))

    result = GraphService(store).analyze_source_mixing(limit=0)

    assert result["total_edge_count"] == 1
    assert result["source_pairs"] == [
        {
            "from_source": "max",
            "to_source": "presence",
            "edge_count": 1,
            "relation_counts": {"relates_to": 1},
            "example_edge_ids": [],
        }
    ]


def test_analyze_source_mixing_handles_empty_and_missing_sources(store: Store):
    empty = GraphService(store).analyze_source_mixing()
    assert empty == {
        "total_edge_count": 0,
        "same_source_edge_count": 0,
        "cross_source_edge_count": 0,
        "cross_source_ratio": 0.0,
        "source_pairs": [],
    }

    store.insert_unit(_unit("unit-a", "Alpha", source_project=SourceProject.MAX))
    store.insert_unit(_unit("unit-b", "Beta", source_project="blank-source"))
    store.insert_unit(_unit("unit-c", "Gamma", source_project="also-blank"))
    store.conn.execute(
        "UPDATE knowledge_units SET source_project = '' WHERE id IN (?, ?)",
        ("unit-b", "unit-c"),
    )
    store.conn.commit()
    store.insert_edge(_edge("edge-a", "unit-a", "unit-b"))
    store.insert_edge(_edge("edge-b", "unit-b", "unit-c"))

    result = GraphService(store).analyze_source_mixing()

    assert result["total_edge_count"] == 2
    assert result["same_source_edge_count"] == 1
    assert result["cross_source_edge_count"] == 1
    assert result["source_pairs"] == [
        {
            "from_source": "max",
            "to_source": "unknown",
            "edge_count": 1,
            "relation_counts": {"relates_to": 1},
            "example_edge_ids": ["edge-a"],
        },
        {
            "from_source": "unknown",
            "to_source": "unknown",
            "edge_count": 1,
            "relation_counts": {"relates_to": 1},
            "example_edge_ids": ["edge-b"],
        },
    ]


@pytest.mark.parametrize("limit", [-1, "bad", None, True])
def test_analyze_source_mixing_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_source_mixing(limit=limit)
