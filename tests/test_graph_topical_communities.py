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


def _unit(unit_id: str, title: str, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        tags=tags,
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def _insert_topical_bridge_graph(store: Store) -> None:
    for unit in [
        _unit("unit-a", "Solar planning", ["energy", "solar"]),
        _unit("unit-b", "Battery storage", ["energy", "storage"]),
        _unit("unit-c", "Wind grid", ["energy", "grid"]),
        _unit("unit-d", "Draft outline", ["writing", "drafting"]),
        _unit("unit-e", "Editing workflow", ["writing", "editing"]),
        _unit("unit-f", "Publishing checklist", ["writing", "publishing"]),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-a-b", "unit-a", "unit-b"),
        _edge("edge-a-c", "unit-a", "unit-c"),
        _edge("edge-b-c", "unit-b", "unit-c"),
        _edge("edge-d-e", "unit-d", "unit-e"),
        _edge("edge-d-f", "unit-d", "unit-f"),
        _edge("edge-e-f", "unit-e", "unit-f"),
        _edge("edge-c-d", "unit-c", "unit-d"),
    ]:
        store.insert_edge(edge)


def test_topical_communities_splits_dense_groups_and_reports_bridge(store: Store):
    _insert_topical_bridge_graph(store)

    result = GraphService(store).topical_communities(representative_limit=3)

    assert result["node_count"] == 6
    assert result["edge_count"] == 7
    assert result["community_count"] == 2
    assert [community["unit_ids"] for community in result["communities"]] == [
        ["unit-a", "unit-b", "unit-c"],
        ["unit-d", "unit-e", "unit-f"],
    ]
    assert [
        (
            community["size"],
            community["representative_tags"],
            community["internal_edge_count"],
            community["density"],
        )
        for community in result["communities"]
    ] == [
        (3, ["energy", "grid", "solar"], 3, 1.0),
        (3, ["writing", "drafting", "editing"], 3, 1.0),
    ]
    assert result["communities"][0]["representative_terms"] == [
        "battery",
        "grid",
        "planning",
    ]
    assert result["bridge_units"] == [
        {
            "unit_id": "unit-c",
            "community_id": result["communities"][0]["community_id"],
            "connected_community_ids": sorted(
                [
                    result["communities"][0]["community_id"],
                    result["communities"][1]["community_id"],
                ]
            ),
            "cross_community_edge_count": 1,
        },
        {
            "unit_id": "unit-d",
            "community_id": result["communities"][1]["community_id"],
            "connected_community_ids": sorted(
                [
                    result["communities"][0]["community_id"],
                    result["communities"][1]["community_id"],
                ]
            ),
            "cross_community_edge_count": 1,
        },
    ]
    assert result["cross_community_edges"] == [
        {
            "from_unit_id": "unit-c",
            "to_unit_id": "unit-d",
            "from_community_id": result["communities"][0]["community_id"],
            "to_community_id": result["communities"][1]["community_id"],
            "edge_id": "edge-c-d",
            "relation": "relates_to",
        }
    ]


def test_topical_communities_limits_results_deterministically(store: Store):
    _insert_topical_bridge_graph(store)

    result = GraphService(store).topical_communities(limit=1, bridge_limit=0)

    assert result["community_count"] == 2
    assert [community["unit_ids"] for community in result["communities"]] == [
        ["unit-a", "unit-b", "unit-c"]
    ]
    assert result["bridge_units"] == []
    assert result["cross_community_edges"] == []


def test_topical_communities_handles_empty_graph(store: Store):
    assert GraphService(store).topical_communities() == {
        "node_count": 0,
        "edge_count": 0,
        "community_count": 0,
        "communities": [],
        "bridge_units": [],
        "cross_community_edges": [],
    }
