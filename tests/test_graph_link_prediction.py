from __future__ import annotations

import math
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


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
) -> KnowledgeEdge:
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


def test_suggest_missing_links_closes_triangles_with_unit_payloads(store: Store):
    _populate(
        store,
        ["unit-hub", "unit-left", "unit-right"],
        [("unit-left", "unit-hub"), ("unit-hub", "unit-right")],
    )

    result = GraphService(store).suggest_missing_links()

    assert len(result) == 1
    candidate = result[0]
    assert candidate["unit_ids"] == ["unit-left", "unit-right"]
    assert [unit["id"] for unit in candidate["units"]] == ["unit-left", "unit-right"]
    assert candidate["score"] == 2.0
    assert candidate["common_neighbor_count"] == 1
    assert [unit["id"] for unit in candidate["common_neighbors"]] == ["unit-hub"]
    assert candidate["common_neighbors"][0]["source_project"] == "max"
    assert candidate["common_neighbors"][0]["source_id"] == "source-unit-hub"
    assert candidate["common_neighbors"][0]["content_type"] == "insight"


def test_suggest_missing_links_filters_existing_edges_self_links_and_isolated_nodes(
    store: Store,
):
    _populate(
        store,
        ["unit-a", "unit-b", "unit-c", "unit-isolated"],
        [
            ("unit-a", "unit-b"),
            ("unit-b", "unit-c"),
            ("unit-c", "unit-a"),
            ("unit-a", "unit-a"),
        ],
    )

    assert GraphService(store).suggest_missing_links() == []


def test_suggest_missing_links_applies_limit_and_min_score(store: Store):
    _populate(
        store,
        ["unit-a", "unit-b", "unit-c", "unit-d", "unit-e"],
        [
            ("unit-a", "unit-c"),
            ("unit-a", "unit-d"),
            ("unit-b", "unit-c"),
            ("unit-b", "unit-d"),
            ("unit-c", "unit-d"),
            ("unit-a", "unit-e"),
        ],
    )

    result = GraphService(store).suggest_missing_links(limit=1, min_score=2.0)

    assert [candidate["unit_ids"] for candidate in result] == [["unit-a", "unit-b"]]
    assert result[0]["common_neighbor_count"] == 2


def test_suggest_missing_links_is_deterministic_for_ties(store: Store):
    _populate(
        store,
        ["unit-c", "unit-a", "unit-d", "unit-b"],
        [
            ("unit-a", "unit-c"),
            ("unit-b", "unit-c"),
            ("unit-a", "unit-d"),
            ("unit-b", "unit-d"),
        ],
    )

    first = GraphService(store).suggest_missing_links()

    reversed_store_fd, reversed_store_path = tempfile.mkstemp(suffix=".db")
    os.close(reversed_store_fd)
    reversed_store = Store(reversed_store_path)
    try:
        _populate(
            reversed_store,
            ["unit-b", "unit-d", "unit-a", "unit-c"],
            [
                ("unit-b", "unit-d"),
                ("unit-a", "unit-d"),
                ("unit-b", "unit-c"),
                ("unit-a", "unit-c"),
            ],
        )
        second = GraphService(reversed_store).suggest_missing_links()
    finally:
        reversed_store.close()
        os.unlink(reversed_store_path)

    comparable_first = [
        {
            "unit_ids": candidate["unit_ids"],
            "score": candidate["score"],
            "common_neighbor_count": candidate["common_neighbor_count"],
            "common_neighbor_ids": [
                unit["id"] for unit in candidate["common_neighbors"]
            ],
        }
        for candidate in first
    ]
    comparable_second = [
        {
            "unit_ids": candidate["unit_ids"],
            "score": candidate["score"],
            "common_neighbor_count": candidate["common_neighbor_count"],
            "common_neighbor_ids": [
                unit["id"] for unit in candidate["common_neighbors"]
            ],
        }
        for candidate in second
    ]

    assert comparable_first == comparable_second
    assert [candidate["unit_ids"] for candidate in first] == [
        ["unit-a", "unit-b"],
        ["unit-c", "unit-d"],
    ]


@pytest.mark.parametrize("limit", [-1, "2", True])
def test_suggest_missing_links_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).suggest_missing_links(limit=limit)


@pytest.mark.parametrize("min_score", [-0.1, "bad", math.inf])
def test_suggest_missing_links_validates_min_score(store: Store, min_score):
    with pytest.raises(ValueError, match="min_score must be a non-negative number"):
        GraphService(store).suggest_missing_links(min_score=min_score)
