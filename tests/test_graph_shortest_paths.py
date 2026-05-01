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


def _unit(unit_id: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=unit_id.removeprefix("unit-").title(),
        content=f"Content for {unit_id}",
        content_type=ContentType.INSIGHT,
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
    weight: float = 1.0,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=weight,
        source=EdgeSource.INFERRED,
    )


def _populate(store: Store) -> None:
    for unit_id in ["unit-a", "unit-b", "unit-c", "unit-d", "unit-isolated"]:
        store.insert_unit(_unit(unit_id))
    for edge in [
        _edge("edge-a-d", "unit-a", "unit-d", EdgeRelation.CHALLENGES, 3.5),
        _edge("edge-a-b", "unit-a", "unit-b", EdgeRelation.BUILDS_ON, 0.75),
        _edge("edge-b-d", "unit-b", "unit-d", EdgeRelation.BUILDS_ON, 1.25),
        _edge("edge-a-c", "unit-a", "unit-c", EdgeRelation.BUILDS_ON, 1.5),
        _edge("edge-c-d", "unit-c", "unit-d", EdgeRelation.BUILDS_ON, 2.5),
    ]:
        store.insert_edge(edge)


def test_shortest_path_between_returns_deterministic_path_explanation(store: Store):
    _populate(store)

    result = GraphService(store).shortest_path_between("unit-a", "unit-d")

    assert result == [
        {
            "unit_ids": ["unit-a", "unit-d"],
            "edge_ids": ["edge-a-d"],
            "relations": ["challenges"],
            "edges": [
                {
                    "id": "edge-a-d",
                    "from_unit_id": "unit-a",
                    "to_unit_id": "unit-d",
                    "relation": "challenges",
                    "weight": 3.5,
                    "source": "inferred",
                    "traversal_from_unit_id": "unit-a",
                    "traversal_to_unit_id": "unit-d",
                    "traversal_direction": "forward",
                }
            ],
            "hop_count": 1,
            "total_weight": 3.5,
        }
    ]


def test_shortest_path_between_relation_filter_changes_path(store: Store):
    _populate(store)

    result = GraphService(store).shortest_path_between(
        "unit-a",
        "unit-d",
        relation="builds_on",
        max_paths=2,
    )

    assert [path["unit_ids"] for path in result] == [
        ["unit-a", "unit-b", "unit-d"],
        ["unit-a", "unit-c", "unit-d"],
    ]
    assert result[0]["edge_ids"] == ["edge-a-b", "edge-b-d"]
    assert result[0]["relations"] == ["builds_on", "builds_on"]
    assert result[0]["hop_count"] == 2
    assert result[0]["total_weight"] == 2.0
    assert result[1]["edge_ids"] == ["edge-a-c", "edge-c-d"]
    assert result[1]["total_weight"] == 4.0


def test_shortest_path_between_relation_filter_returns_empty_when_no_path(
    store: Store,
):
    _populate(store)

    assert (
        GraphService(store).shortest_path_between(
            "unit-a",
            "unit-d",
            relation="relates_to",
        )
        == []
    )


def test_shortest_path_between_reports_reverse_traversal(store: Store):
    _populate(store)

    result = GraphService(store).shortest_path_between(
        "unit-d",
        "unit-a",
        relation="challenges",
    )

    assert result[0]["unit_ids"] == ["unit-d", "unit-a"]
    assert result[0]["edges"][0]["from_unit_id"] == "unit-a"
    assert result[0]["edges"][0]["to_unit_id"] == "unit-d"
    assert result[0]["edges"][0]["traversal_direction"] == "reverse"


def test_shortest_path_between_returns_empty_for_disconnected_units(store: Store):
    _populate(store)

    assert GraphService(store).shortest_path_between("unit-a", "unit-isolated") == []


@pytest.mark.parametrize(
    ("source_unit_id", "target_unit_id", "match"),
    [
        ("missing-source", "unit-a", "source_unit_id not found: missing-source"),
        ("unit-a", "missing-target", "target_unit_id not found: missing-target"),
    ],
)
def test_shortest_path_between_validates_missing_units(
    store: Store,
    source_unit_id: str,
    target_unit_id: str,
    match: str,
):
    _populate(store)

    with pytest.raises(ValueError, match=match):
        GraphService(store).shortest_path_between(source_unit_id, target_unit_id)


@pytest.mark.parametrize("max_paths", [-1, "2", True])
def test_shortest_path_between_validates_max_paths(store: Store, max_paths):
    _populate(store)

    with pytest.raises(ValueError, match="max_paths must be a non-negative integer"):
        GraphService(store).shortest_path_between(
            "unit-a",
            "unit-d",
            max_paths=max_paths,
        )


def test_shortest_path_between_accepts_zero_max_paths(store: Store):
    _populate(store)

    assert (
        GraphService(store).shortest_path_between(
            "unit-a",
            "unit-d",
            max_paths=0,
        )
        == []
    )
