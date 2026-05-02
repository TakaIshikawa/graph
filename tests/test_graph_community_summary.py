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


def _unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def _insert_units(store: Store, units: list[tuple[str, str]]) -> None:
    for unit_id, title in units:
        store.insert_unit(_unit(unit_id, title))


def _insert_edges(store: Store, edges: list[tuple[str, str, str]]) -> None:
    for edge_id, from_unit_id, to_unit_id in edges:
        store.insert_edge(_edge(edge_id, from_unit_id, to_unit_id))


def test_summarize_communities_returns_two_disconnected_clusters(store: Store):
    _insert_units(
        store,
        [
            ("unit-a", "Alpha"),
            ("unit-b", "Beta"),
            ("unit-c", "Gamma"),
            ("unit-d", "Delta"),
            ("unit-e", "Epsilon"),
        ],
    )
    _insert_edges(
        store,
        [
            ("edge-a-b", "unit-a", "unit-b"),
            ("edge-b-c", "unit-b", "unit-c"),
            ("edge-a-c", "unit-a", "unit-c"),
            ("edge-d-e", "unit-d", "unit-e"),
        ],
    )

    result = GraphService(store).summarize_communities()

    assert result == {
        "node_count": 5,
        "edge_count": 4,
        "community_count": 2,
        "communities": [
            {
                "community_id": result["communities"][0]["community_id"],
                "member_ids": ["unit-a", "unit-b", "unit-c"],
                "size": 3,
                "representative_labels": ["Alpha", "Beta", "Gamma"],
                "internal_edge_count": 3,
                "possible_edge_count": 3,
                "density": 1.0,
            },
            {
                "community_id": result["communities"][1]["community_id"],
                "member_ids": ["unit-d", "unit-e"],
                "size": 2,
                "representative_labels": ["Delta", "Epsilon"],
                "internal_edge_count": 1,
                "possible_edge_count": 1,
                "density": 1.0,
            },
        ],
    }
    assert result["communities"][0]["community_id"].startswith("community-")
    assert result["communities"][1]["community_id"].startswith("community-")


def test_summarize_communities_splits_dense_clusters_connected_by_bridge(store: Store):
    _insert_units(
        store,
        [
            ("unit-a", "Alpha"),
            ("unit-b", "Beta"),
            ("unit-c", "Gamma"),
            ("unit-d", "Delta"),
            ("unit-e", "Epsilon"),
            ("unit-f", "Zeta"),
        ],
    )
    _insert_edges(
        store,
        [
            ("edge-a-b", "unit-a", "unit-b"),
            ("edge-b-c", "unit-b", "unit-c"),
            ("edge-a-c", "unit-a", "unit-c"),
            ("edge-d-e", "unit-d", "unit-e"),
            ("edge-e-f", "unit-e", "unit-f"),
            ("edge-d-f", "unit-d", "unit-f"),
            ("edge-c-d", "unit-c", "unit-d"),
        ],
    )

    result = GraphService(store).summarize_communities()

    assert [community["member_ids"] for community in result["communities"]] == [
        ["unit-a", "unit-b", "unit-c"],
        ["unit-d", "unit-e", "unit-f"],
    ]
    assert [community["internal_edge_count"] for community in result["communities"]] == [3, 3]
    assert [community["density"] for community in result["communities"]] == [1.0, 1.0]


def test_summarize_communities_handles_singletons_and_empty_graph(store: Store):
    assert GraphService(store).summarize_communities() == {
        "node_count": 0,
        "edge_count": 0,
        "community_count": 0,
        "communities": [],
    }

    store.insert_unit(_unit("unit-a", ""))

    assert GraphService(store).summarize_communities() == {
        "node_count": 1,
        "edge_count": 0,
        "community_count": 1,
        "communities": [
            {
                "community_id": GraphService(store).summarize_communities()["communities"][0][
                    "community_id"
                ],
                "member_ids": ["unit-a"],
                "size": 1,
                "representative_labels": ["unit-a"],
                "internal_edge_count": 0,
                "possible_edge_count": 0,
                "density": 0,
            }
        ],
    }


def test_summarize_communities_has_deterministic_ordering(store: Store):
    _insert_units(
        store,
        [
            ("unit-d", "Delta"),
            ("unit-c", "Gamma"),
            ("unit-b", "Beta"),
            ("unit-a", "Alpha"),
        ],
    )
    _insert_edges(
        store,
        [
            ("edge-c-d", "unit-c", "unit-d"),
            ("edge-a-b", "unit-a", "unit-b"),
        ],
    )

    first = GraphService(store).summarize_communities()
    second = GraphService(store).get_community_summary()
    third = GraphService(store).analyze_community_summary()

    assert first == second == third
    assert [community["member_ids"] for community in first["communities"]] == [
        ["unit-a", "unit-b"],
        ["unit-c", "unit-d"],
    ]


@pytest.mark.parametrize("limit", [-1, "bad", True])
def test_summarize_communities_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        GraphService(store).summarize_communities(limit=limit)
