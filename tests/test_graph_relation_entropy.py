from __future__ import annotations

import os
import tempfile

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
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
    content_type: ContentType = ContentType.INSIGHT,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=content_type,
        tags=tags or [],
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
    )


def test_analyze_relation_entropy_empty_graph(store: Store):
    assert GraphService(store).analyze_relation_entropy() == {
        "group_by": "source_project",
        "limit": 20,
        "total_edges": 0,
        "global_relation_counts": {},
        "group_count": 0,
        "groups": [],
    }


def test_analyze_relation_entropy_validates_group_by_and_limit(store: Store):
    service = GraphService(store)

    with pytest.raises(
        ValueError, match="group_by must be source_project, tag, or content_type"
    ):
        service.analyze_relation_entropy("source_entity_type")

    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        service.analyze_relation_entropy(limit=-1)


def test_analyze_relation_entropy_groups_by_source_project_with_dominance(
    store: Store,
):
    for unit in [
        _unit("unit-max-a", "Max A", source_project=SourceProject.MAX),
        _unit("unit-max-b", "Max B", source_project=SourceProject.MAX),
        _unit("unit-presence", "Presence", source_project=SourceProject.PRESENCE),
        _unit("unit-forty-two", "Forty Two", source_project=SourceProject.FORTY_TWO),
        _unit("unit-me-a", "Me A", source_project=SourceProject.ME),
        _unit("unit-me-b", "Me B", source_project=SourceProject.ME),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-1", "unit-max-a", "unit-max-b", EdgeRelation.BUILDS_ON),
        _edge("edge-2", "unit-max-a", "unit-presence", EdgeRelation.BUILDS_ON),
        _edge("edge-3", "unit-presence", "unit-max-a", EdgeRelation.INSPIRES),
        _edge("edge-4", "unit-forty-two", "unit-max-a", EdgeRelation.REFERENCES),
        _edge("edge-5", "unit-me-a", "unit-me-b", EdgeRelation.RELATES_TO),
        _edge("edge-6", "unit-me-b", "unit-me-a", EdgeRelation.RELATES_TO),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_relation_entropy(limit=3)

    assert result["group_by"] == "source_project"
    assert result["limit"] == 3
    assert result["total_edges"] == 6
    assert result["global_relation_counts"] == {
        "builds_on": 2,
        "inspires": 1,
        "references": 1,
        "relates_to": 2,
    }
    assert result["group_count"] == 4
    assert result["groups"] == [
        {
            "group": "me",
            "relation_counts": {"relates_to": 2},
            "dominant_relation": "relates_to",
            "dominant_relation_share": 1.0,
            "entropy": 0.0,
            "total_edges": 2,
        },
        {
            "group": "forty_two",
            "relation_counts": {"references": 1},
            "dominant_relation": "references",
            "dominant_relation_share": 1.0,
            "entropy": 0.0,
            "total_edges": 1,
        },
        {
            "group": "presence",
            "relation_counts": {"builds_on": 1, "inspires": 1},
            "dominant_relation": "builds_on",
            "dominant_relation_share": 0.5,
            "entropy": 1.0,
            "total_edges": 2,
        },
    ]


def test_analyze_relation_entropy_groups_by_tag_and_sorts_ties(store: Store):
    for unit in [
        _unit("unit-a", "A", tags=["energy", "solar"]),
        _unit("unit-b", "B", tags=["energy"]),
        _unit("unit-c", "C", tags=["writing"]),
        _unit("unit-d", "D"),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-1", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
        _edge("edge-2", "unit-a", "unit-c", EdgeRelation.BUILDS_ON),
        _edge("edge-3", "unit-c", "unit-a", EdgeRelation.INSPIRES),
        _edge("edge-4", "unit-d", "unit-d", EdgeRelation.REFERENCES),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_relation_entropy("tag", limit=2)

    assert result["total_edges"] == 4
    assert result["global_relation_counts"] == {
        "builds_on": 2,
        "inspires": 1,
        "references": 1,
    }
    assert result["group_count"] == 3
    assert [group["group"] for group in result["groups"]] == ["energy", "solar"]
    assert result["groups"][0] == {
        "group": "energy",
        "relation_counts": {"builds_on": 2, "inspires": 1},
        "dominant_relation": "builds_on",
        "dominant_relation_share": pytest.approx(2 / 3),
        "entropy": pytest.approx(0.9182958340544896),
        "total_edges": 3,
    }
    assert result["groups"][1]["relation_counts"] == {"builds_on": 2, "inspires": 1}


def test_analyze_relation_entropy_groups_by_content_type(store: Store):
    for unit in [
        _unit("unit-insight", "Insight", content_type=ContentType.INSIGHT),
        _unit("unit-finding", "Finding", content_type=ContentType.FINDING),
        _unit("unit-artifact", "Artifact", content_type=ContentType.ARTIFACT),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-1", "unit-insight", "unit-finding", EdgeRelation.CHALLENGES),
        _edge("edge-2", "unit-finding", "unit-artifact", EdgeRelation.CHALLENGES),
        _edge("edge-3", "unit-artifact", "unit-insight", EdgeRelation.REFINES),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_relation_entropy("content_type")

    assert result["group_count"] == 3
    assert result["groups"][0] == {
        "group": "finding",
        "relation_counts": {"challenges": 2},
        "dominant_relation": "challenges",
        "dominant_relation_share": 1.0,
        "entropy": 0.0,
        "total_edges": 2,
    }
