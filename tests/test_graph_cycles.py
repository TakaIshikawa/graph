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
    content_type: ContentType = ContentType.INSIGHT,
) -> KnowledgeUnit:
    return store.insert_unit(
        KnowledgeUnit(
            id=unit_id,
            source_project=source_project,
            source_id=unit_id,
            source_entity_type="insight",
            title=title,
            content=f"{title} content",
            content_type=content_type,
        )
    )


def _insert_edge(store: Store, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return store.insert_edge(
        KnowledgeEdge(
            from_unit_id=from_unit_id,
            to_unit_id=to_unit_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.MANUAL,
        )
    )


def test_analyze_cycles_empty_graph_returns_zero_counts(store: Store):
    result = GraphService(store).analyze_cycles()

    assert result == {
        "stats": {
            "node_count": 0,
            "edge_count": 0,
            "cycle_count": 0,
            "returned_count": 0,
            "limit": 20,
            "max_length": None,
        },
        "cycles": [],
    }


def test_analyze_cycles_acyclic_graph_returns_empty_cycle_list(store: Store):
    _insert_unit(store, "unit-a", "A")
    _insert_unit(store, "unit-b", "B")
    _insert_unit(store, "unit-c", "C")
    _insert_edge(store, "unit-a", "unit-b")
    _insert_edge(store, "unit-b", "unit-c")

    result = GraphService(store).analyze_cycles()

    assert result["stats"]["node_count"] == 3
    assert result["stats"]["edge_count"] == 2
    assert result["stats"]["cycle_count"] == 0
    assert result["stats"]["returned_count"] == 0
    assert result["cycles"] == []


def test_analyze_cycles_returns_multiple_cycles_with_lightweight_units(store: Store):
    _insert_unit(store, "unit-a", "Alpha")
    _insert_unit(
        store,
        "unit-b",
        "Beta",
        source_project=SourceProject.FORTY_TWO,
        content_type=ContentType.FINDING,
    )
    _insert_unit(store, "unit-c", "Gamma")
    _insert_unit(store, "unit-d", "Delta")
    for from_id, to_id in [
        ("unit-b", "unit-a"),
        ("unit-a", "unit-b"),
        ("unit-a", "unit-c"),
        ("unit-c", "unit-d"),
        ("unit-d", "unit-a"),
    ]:
        _insert_edge(store, from_id, to_id)

    result = GraphService(store).analyze_cycles(limit=10)

    assert result["stats"] == {
        "node_count": 4,
        "edge_count": 5,
        "cycle_count": 2,
        "returned_count": 2,
        "limit": 10,
        "max_length": None,
    }
    assert [cycle["unit_ids"] for cycle in result["cycles"]] == [
        ["unit-a", "unit-b"],
        ["unit-a", "unit-c", "unit-d"],
    ]
    assert result["cycles"][0] == {
        "unit_ids": ["unit-a", "unit-b"],
        "length": 2,
        "units": [
            {
                "id": "unit-a",
                "title": "Alpha",
                "source_project": "max",
                "content_type": "insight",
            },
            {
                "id": "unit-b",
                "title": "Beta",
                "source_project": "forty_two",
                "content_type": "finding",
            },
        ],
    }


def test_analyze_cycles_order_is_deterministic_across_edge_insert_order(store: Store):
    for unit_id in ["unit-a", "unit-b", "unit-c", "unit-d"]:
        _insert_unit(store, unit_id, unit_id)
    for from_id, to_id in [
        ("unit-d", "unit-b"),
        ("unit-c", "unit-d"),
        ("unit-b", "unit-c"),
        ("unit-c", "unit-a"),
        ("unit-a", "unit-c"),
    ]:
        _insert_edge(store, from_id, to_id)

    result = GraphService(store).analyze_cycles(limit=10)

    assert [cycle["unit_ids"] for cycle in result["cycles"]] == [
        ["unit-a", "unit-c"],
        ["unit-b", "unit-c", "unit-d"],
    ]


def test_analyze_cycles_applies_limit_after_counting_matching_cycles(store: Store):
    for unit_id in ["unit-a", "unit-b", "unit-c", "unit-d"]:
        _insert_unit(store, unit_id, unit_id)
    for from_id, to_id in [
        ("unit-a", "unit-b"),
        ("unit-b", "unit-a"),
        ("unit-c", "unit-d"),
        ("unit-d", "unit-c"),
    ]:
        _insert_edge(store, from_id, to_id)

    result = GraphService(store).analyze_cycles(limit=1)

    assert result["stats"]["cycle_count"] == 2
    assert result["stats"]["returned_count"] == 1
    assert result["cycles"][0]["unit_ids"] == ["unit-a", "unit-b"]


def test_analyze_cycles_accepts_zero_limit(store: Store):
    _insert_unit(store, "unit-a", "A")
    _insert_unit(store, "unit-b", "B")
    _insert_edge(store, "unit-a", "unit-b")
    _insert_edge(store, "unit-b", "unit-a")

    result = GraphService(store).analyze_cycles(limit=0)

    assert result["stats"]["cycle_count"] == 1
    assert result["stats"]["returned_count"] == 0
    assert result["cycles"] == []


def test_analyze_cycles_filters_by_max_length(store: Store):
    for unit_id in ["unit-a", "unit-b", "unit-c", "unit-d"]:
        _insert_unit(store, unit_id, unit_id)
    for from_id, to_id in [
        ("unit-a", "unit-b"),
        ("unit-b", "unit-a"),
        ("unit-a", "unit-c"),
        ("unit-c", "unit-d"),
        ("unit-d", "unit-a"),
    ]:
        _insert_edge(store, from_id, to_id)

    result = GraphService(store).analyze_cycles(limit=10, max_length=2)

    assert result["stats"]["cycle_count"] == 1
    assert result["stats"]["max_length"] == 2
    assert [cycle["unit_ids"] for cycle in result["cycles"]] == [
        ["unit-a", "unit-b"]
    ]


@pytest.mark.parametrize("limit", [-1, 1.5, "2", True, None])
def test_analyze_cycles_rejects_invalid_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_cycles(limit=limit)


@pytest.mark.parametrize("max_length", [0, -1, 1.5, "2", True])
def test_analyze_cycles_rejects_invalid_max_length(store: Store, max_length):
    with pytest.raises(ValueError, match="max_length must be a positive integer or None"):
        GraphService(store).analyze_cycles(max_length=max_length)
