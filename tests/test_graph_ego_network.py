from __future__ import annotations

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    yield store
    store.close()


def _unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="insight",
        title=title,
        content=f"Content for {title}",
    )


def _edge(from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


@pytest.fixture
def ego_store(store: Store):
    for unit_id, title in [
        ("unit-a", "A"),
        ("unit-b", "B"),
        ("unit-c", "C"),
        ("unit-d", "D"),
        ("unit-e", "E"),
        ("unit-f", "F"),
    ]:
        store.insert_unit(_unit(unit_id, title))

    for from_unit_id, to_unit_id in [
        ("unit-a", "unit-b"),
        ("unit-b", "unit-c"),
        ("unit-d", "unit-b"),
        ("unit-c", "unit-d"),
        ("unit-e", "unit-f"),
    ]:
        store.insert_edge(_edge(from_unit_id, to_unit_id))

    return store


def _node_ids(payload: dict) -> list[str]:
    return [node["id"] for node in payload["nodes"]]


def _edge_pairs(payload: dict) -> list[tuple[str, str]]:
    return [(edge["from_unit_id"], edge["to_unit_id"]) for edge in payload["edges"]]


def test_ego_network_radius_one_returns_direct_undirected_neighbors_and_induced_edges(
    ego_store: Store,
):
    payload = GraphService(ego_store).ego_network("unit-b")

    assert payload["center_unit_id"] == "unit-b"
    assert payload["radius"] == 1
    assert payload["undirected"] is True
    assert _node_ids(payload) == ["unit-a", "unit-b", "unit-c", "unit-d"]
    assert _edge_pairs(payload) == [
        ("unit-a", "unit-b"),
        ("unit-b", "unit-c"),
        ("unit-c", "unit-d"),
        ("unit-d", "unit-b"),
    ]
    assert payload["summary"] == {"node_count": 4, "edge_count": 4}


def test_ego_network_radius_two_includes_two_hop_neighbors_only(ego_store: Store):
    payload = GraphService(ego_store).ego_network("unit-a", radius=2)

    assert _node_ids(payload) == ["unit-a", "unit-b", "unit-c", "unit-d"]
    assert _edge_pairs(payload) == [
        ("unit-a", "unit-b"),
        ("unit-b", "unit-c"),
        ("unit-c", "unit-d"),
        ("unit-d", "unit-b"),
    ]
    assert "unit-e" not in _node_ids(payload)
    assert ("unit-e", "unit-f") not in _edge_pairs(payload)


def test_ego_network_directed_mode_follows_outgoing_edges(ego_store: Store):
    payload = GraphService(ego_store).ego_network(
        "unit-b",
        radius=1,
        undirected=False,
    )

    assert _node_ids(payload) == ["unit-b", "unit-c"]
    assert _edge_pairs(payload) == [("unit-b", "unit-c")]
    assert payload["summary"] == {"node_count": 2, "edge_count": 1}


def test_ego_network_rejects_unknown_units(ego_store: Store):
    with pytest.raises(KeyError, match="Unit not found: missing"):
        GraphService(ego_store).ego_network("missing")


@pytest.mark.parametrize("radius", [0, -1, 1.5, True])
def test_ego_network_validates_radius(ego_store: Store, radius):
    with pytest.raises(ValueError, match="radius must be a positive integer"):
        GraphService(ego_store).ego_network("unit-a", radius=radius)
