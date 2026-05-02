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
        content=f"Content for {title}",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
    *,
    source: EdgeSource = EdgeSource.MANUAL,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=source,
        metadata=metadata or {},
    )


@pytest.fixture
def leaf_store(store: Store):
    for unit in [
        _unit("unit-hub", "Hub", tags=["systems"]),
        _unit("unit-outgoing-leaf", "Alpha Leaf", tags=["draft"]),
        _unit(
            "unit-incoming-leaf",
            "Beta Leaf",
            source_project=SourceProject.PRESENCE,
            tags=["archive", "topic"],
        ),
        _unit("unit-bridge-a", "Bridge A"),
        _unit("unit-bridge-b", "Bridge B"),
        _unit("unit-isolated", "Isolated"),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge(
            "edge-outgoing-leaf",
            "unit-outgoing-leaf",
            "unit-hub",
            EdgeRelation.BUILDS_ON,
            metadata={"confidence": "reviewed"},
        ),
        _edge(
            "edge-incoming-leaf",
            "unit-hub",
            "unit-incoming-leaf",
            EdgeRelation.REFERENCES,
            source=EdgeSource.INFERRED,
        ),
        _edge(
            "edge-bridge-a-b",
            "unit-bridge-a",
            "unit-bridge-b",
            EdgeRelation.RELATES_TO,
        ),
        _edge(
            "edge-bridge-b-a",
            "unit-bridge-b",
            "unit-bridge-a",
            EdgeRelation.RELATES_TO,
        ),
    ]:
        store.insert_edge(edge)

    return store


def test_analyze_leaf_nodes_returns_leaf_neighbor_and_relationship_metadata(
    leaf_store: Store,
):
    result = GraphService(leaf_store).analyze_leaf_nodes()

    assert [item["unit"]["id"] for item in result] == [
        "unit-outgoing-leaf",
        "unit-incoming-leaf",
    ]
    assert result[0] == {
        "unit": {
            "id": "unit-outgoing-leaf",
            "source_project": "max",
            "source_id": "source-unit-outgoing-leaf",
            "source_entity_type": "insight",
            "title": "Alpha Leaf",
            "content_type": "insight",
            "tags": ["draft"],
        },
        "neighbor": {
            "id": "unit-hub",
            "source_project": "max",
            "source_id": "source-unit-hub",
            "source_entity_type": "insight",
            "title": "Hub",
            "content_type": "insight",
            "tags": ["systems"],
        },
        "edge": {
            "id": "edge-outgoing-leaf",
            "from_unit_id": "unit-outgoing-leaf",
            "to_unit_id": "unit-hub",
            "relation": "builds_on",
            "weight": 1.0,
            "source": "manual",
            "metadata": {"confidence": "reviewed"},
            "created_at": result[0]["edge"]["created_at"],
        },
        "relationship": "builds_on",
        "direction": "outgoing",
        "degree": 1,
        "in_degree": 0,
        "out_degree": 1,
        "reason_code": "single_outgoing_edge",
        "reason": "Only one outgoing relationship connects this unit.",
    }
    assert result[1]["unit"]["source_project"] == "presence"
    assert result[1]["unit"]["tags"] == ["archive", "topic"]
    assert result[1]["neighbor"]["id"] == "unit-hub"
    assert result[1]["relationship"] == "references"
    assert result[1]["edge"]["source"] == "inferred"
    assert result[1]["direction"] == "incoming"
    assert result[1]["in_degree"] == 1
    assert result[1]["out_degree"] == 0
    assert result[1]["reason_code"] == "single_incoming_edge"


def test_analyze_leaf_nodes_uses_directed_incident_degree(leaf_store: Store):
    result = GraphService(leaf_store).analyze_leaf_nodes()

    leaf_ids = {item["unit"]["id"] for item in result}
    assert "unit-bridge-a" not in leaf_ids
    assert "unit-bridge-b" not in leaf_ids
    assert "unit-isolated" not in leaf_ids
    assert "unit-hub" not in leaf_ids


def test_analyze_leaf_nodes_returns_empty_list_when_no_leaves(store: Store):
    for unit in [
        _unit("unit-a", "A"),
        _unit("unit-b", "B"),
        _unit("unit-c", "C"),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-a-b", "unit-a", "unit-b", EdgeRelation.RELATES_TO),
        _edge("edge-b-c", "unit-b", "unit-c", EdgeRelation.RELATES_TO),
        _edge("edge-c-a", "unit-c", "unit-a", EdgeRelation.RELATES_TO),
    ]:
        store.insert_edge(edge)

    assert GraphService(store).analyze_leaf_nodes() == []


def test_analyze_leaf_nodes_sorts_by_source_or_neighbor(leaf_store: Store):
    service = GraphService(leaf_store)

    assert [item["unit"]["id"] for item in service.analyze_leaf_nodes(sort_by="source")] == [
        "unit-outgoing-leaf",
        "unit-incoming-leaf",
    ]
    assert [
        item["unit"]["id"] for item in service.analyze_leaf_nodes(sort_by="neighbor")
    ] == [
        "unit-outgoing-leaf",
        "unit-incoming-leaf",
    ]


def test_analyze_leaf_nodes_validates_sort_by(store: Store):
    with pytest.raises(ValueError, match="sort_by must be one of"):
        GraphService(store).analyze_leaf_nodes(sort_by="degree")
