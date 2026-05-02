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


def _unit(unit_id: str, title: str, content: str | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content or f"Content for {title}",
        content_type=ContentType.INSIGHT,
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
    weight: float = 1.0,
    source: EdgeSource = EdgeSource.INFERRED,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=weight,
        source=source,
    )


def test_shortest_path_explanation_returns_direct_path_with_endpoint_context(
    store: Store,
):
    store.insert_unit(_unit("unit-a", "Alpha", "Alpha explains the starting premise."))
    store.insert_unit(_unit("unit-b", "Beta", "Beta is the target conclusion."))
    store.insert_edge(
        _edge(
            "edge-a-b",
            "unit-a",
            "unit-b",
            EdgeRelation.CHALLENGES,
            2.5,
            EdgeSource.MANUAL,
        )
    )

    result = GraphService(store).build_shortest_path_explanation("unit-a", "unit-b")

    assert result["path_found"] is True
    assert result["unit_ids"] == ["unit-a", "unit-b"]
    assert result["units"] == [
        {
            "id": "unit-a",
            "source_project": "max",
            "source_id": "source-unit-a",
            "source_entity_type": "insight",
            "title": "Alpha",
            "content_type": "insight",
            "content_snippet": "Alpha explains the starting premise.",
        },
        {
            "id": "unit-b",
            "source_project": "max",
            "source_id": "source-unit-b",
            "source_entity_type": "insight",
            "title": "Beta",
            "content_type": "insight",
            "content_snippet": "Beta is the target conclusion.",
        },
    ]
    assert result["relations"] == ["challenges"]
    assert result["relation_labels"] == ["challenges"]
    assert result["hop_count"] == 1
    assert result["total_weight"] == 2.5
    assert result["hops"] == [
        {
            "edge_id": "edge-a-b",
            "from_unit_id": "unit-a",
            "to_unit_id": "unit-b",
            "relation": "challenges",
            "relation_label": "challenges",
            "weight": 2.5,
            "source": "manual",
            "traversal_from_unit_id": "unit-a",
            "traversal_to_unit_id": "unit-b",
            "traversal_direction": "forward",
            "from_unit": result["units"][0],
            "to_unit": result["units"][1],
        }
    ]


def test_shortest_path_explanation_returns_multi_hop_details(store: Store):
    for unit_id, title in [
        ("unit-a", "Alpha"),
        ("unit-b", "Beta"),
        ("unit-c", "Gamma"),
    ]:
        store.insert_unit(_unit(unit_id, title))
    store.insert_edge(
        _edge("edge-a-b", "unit-a", "unit-b", EdgeRelation.BUILDS_ON, 0.75)
    )
    store.insert_edge(
        _edge("edge-b-c", "unit-b", "unit-c", EdgeRelation.INSPIRES, 1.25)
    )

    result = GraphService(store).build_shortest_path_explanation("unit-a", "unit-c")

    assert result["path_found"] is True
    assert result["unit_ids"] == ["unit-a", "unit-b", "unit-c"]
    assert [unit["title"] for unit in result["units"]] == ["Alpha", "Beta", "Gamma"]
    assert result["relations"] == ["builds_on", "inspires"]
    assert result["relation_labels"] == ["builds on", "inspires"]
    assert result["hop_count"] == 2
    assert result["total_weight"] == 2.0
    assert [hop["edge_id"] for hop in result["hops"]] == ["edge-a-b", "edge-b-c"]
    assert [hop["weight"] for hop in result["hops"]] == [0.75, 1.25]
    assert [hop["source"] for hop in result["hops"]] == ["inferred", "inferred"]
    assert result["hops"][0]["from_unit"]["title"] == "Alpha"
    assert result["hops"][0]["to_unit"]["title"] == "Beta"
    assert result["hops"][1]["from_unit"]["title"] == "Beta"
    assert result["hops"][1]["to_unit"]["title"] == "Gamma"


def test_shortest_path_explanation_returns_structured_no_path_payload(store: Store):
    store.insert_unit(_unit("unit-a", "Alpha"))
    store.insert_unit(_unit("unit-isolated", "Isolated"))

    result = GraphService(store).build_shortest_path_explanation(
        "unit-a",
        "unit-isolated",
    )

    assert result == {
        "from_unit_id": "unit-a",
        "to_unit_id": "unit-isolated",
        "path_found": False,
        "unit_ids": [],
        "units": [],
        "hops": [],
        "relations": [],
        "relation_labels": [],
        "hop_count": 0,
        "total_weight": 0.0,
        "message": "No path found between the selected units.",
    }


@pytest.mark.parametrize(
    ("from_unit_id", "to_unit_id", "match"),
    [
        ("missing-source", "unit-a", "source_unit_id not found: missing-source"),
        ("unit-a", "missing-target", "target_unit_id not found: missing-target"),
    ],
)
def test_shortest_path_explanation_validates_unknown_unit_ids(
    store: Store,
    from_unit_id: str,
    to_unit_id: str,
    match: str,
):
    store.insert_unit(_unit("unit-a", "Alpha"))

    with pytest.raises(ValueError, match=match):
        GraphService(store).build_shortest_path_explanation(from_unit_id, to_unit_id)
