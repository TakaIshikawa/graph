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


def _unit(unit_id: str, title: str | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title or unit_id,
        content=f"Content for {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=[unit_id],
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
        source=EdgeSource.INFERRED,
    )


def _populate(store: Store, unit_ids: list[str], edges: list[tuple[str, str]]) -> None:
    for unit_id in unit_ids:
        store.insert_unit(_unit(unit_id, unit_id.removeprefix("unit-").title()))
    for index, (from_unit_id, to_unit_id) in enumerate(edges):
        store.insert_edge(_edge(f"edge-{index}", from_unit_id, to_unit_id))


def test_analyze_articulation_points_orders_by_impact_then_unit_id(store: Store):
    _populate(
        store,
        [
            "unit-a",
            "unit-b",
            "unit-c",
            "unit-d",
            "unit-e",
            "unit-f",
            "unit-g",
            "unit-h",
        ],
        [
            ("unit-a", "unit-b"),
            ("unit-b", "unit-c"),
            ("unit-c", "unit-d"),
            ("unit-c", "unit-e"),
            ("unit-c", "unit-f"),
            ("unit-f", "unit-g"),
            ("unit-g", "unit-h"),
        ],
    )

    result = GraphService(store).analyze_articulation_points()

    assert [candidate["unit_id"] for candidate in result] == [
        "unit-c",
        "unit-f",
        "unit-b",
        "unit-g",
    ]
    assert result[0] == {
        "unit_id": "unit-c",
        "title": "C",
        "source_project": "max",
        "source_id": "source-unit-c",
        "source_entity_type": "insight",
        "content_type": "insight",
        "component_size_impact": 4,
        "neighbor_count": 4,
        "affected_component_sizes": [3, 2, 1, 1],
        "impact": {
            "component_count_before": 1,
            "component_count_after": 4,
            "original_component_size": 8,
            "largest_remaining_component_size": 3,
            "affected_component_sizes": [3, 2, 1, 1],
        },
    }


def test_analyze_articulation_points_handles_disconnected_graphs(store: Store):
    _populate(
        store,
        [
            "unit-a",
            "unit-b",
            "unit-c",
            "unit-x",
            "unit-y",
            "unit-z",
            "unit-isolated",
        ],
        [
            ("unit-a", "unit-b"),
            ("unit-b", "unit-c"),
            ("unit-x", "unit-y"),
            ("unit-y", "unit-z"),
        ],
    )

    result = GraphService(store).analyze_articulation_points()

    assert [candidate["unit_id"] for candidate in result] == ["unit-b", "unit-y"]
    assert [candidate["component_size_impact"] for candidate in result] == [1, 1]
    assert [candidate["affected_component_sizes"] for candidate in result] == [
        [1, 1],
        [1, 1],
    ]
    assert {candidate["impact"]["component_count_before"] for candidate in result} == {3}
    assert {candidate["impact"]["component_count_after"] for candidate in result} == {4}


def test_analyze_articulation_points_reports_no_candidates_for_stable_graphs(
    store: Store,
):
    assert GraphService(store).analyze_articulation_points() == []

    _populate(store, ["unit-a"], [])
    assert GraphService(store).analyze_articulation_points() == []

    _populate(
        store,
        ["unit-b", "unit-c", "unit-d"],
        [
            ("unit-a", "unit-b"),
            ("unit-a", "unit-c"),
            ("unit-a", "unit-d"),
            ("unit-b", "unit-c"),
            ("unit-b", "unit-d"),
            ("unit-c", "unit-d"),
        ],
    )
    assert GraphService(store).analyze_articulation_points() == []


def test_analyze_articulation_points_applies_limit(store: Store):
    _populate(
        store,
        ["unit-a", "unit-b", "unit-c", "unit-d", "unit-e"],
        [
            ("unit-a", "unit-b"),
            ("unit-b", "unit-c"),
            ("unit-c", "unit-d"),
            ("unit-d", "unit-e"),
        ],
    )

    result = GraphService(store).analyze_articulation_points(limit=2)

    assert [candidate["unit_id"] for candidate in result] == ["unit-c", "unit-b"]


@pytest.mark.parametrize("limit", [-1, "bad"])
def test_analyze_articulation_points_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_articulation_points(limit=limit)


def test_analyze_articulation_points_accepts_zero_limit(store: Store):
    _populate(store, ["unit-a", "unit-b", "unit-c"], [("unit-a", "unit-b"), ("unit-b", "unit-c")])

    assert GraphService(store).analyze_articulation_points(limit=0) == []
