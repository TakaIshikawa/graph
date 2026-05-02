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
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=content_type,
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


def test_analyze_relation_imbalance_groups_by_source_project_with_mixed_relations(
    store: Store,
):
    for unit in [
        _unit("unit-max-a", "Max A", source_project=SourceProject.MAX),
        _unit("unit-max-b", "Max B", source_project=SourceProject.MAX),
        _unit("unit-presence-a", "Presence A", source_project=SourceProject.PRESENCE),
        _unit("unit-presence-b", "Presence B", source_project=SourceProject.PRESENCE),
        _unit("unit-forty-two", "Forty Two", source_project=SourceProject.FORTY_TWO),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-1", "unit-max-a", "unit-presence-a", EdgeRelation.INSPIRES),
        _edge("edge-2", "unit-presence-b", "unit-max-b", EdgeRelation.INSPIRES),
        _edge("edge-3", "unit-max-a", "unit-presence-b", EdgeRelation.BUILDS_ON),
        _edge("edge-4", "unit-presence-a", "unit-max-a", EdgeRelation.REFERENCES),
        _edge("edge-5", "unit-max-b", "unit-forty-two", EdgeRelation.RELATES_TO),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_relation_imbalance()

    assert result["group_by"] == "source_project"
    assert result["pair_count"] == 2
    assert result["pairs"] == [
        {
            "groups": ["forty_two", "max"],
            "relation_counts": {"relates_to": 1},
            "dominant_relation": "relates_to",
            "total_edges": 1,
            "imbalance_score": 1.0,
        },
        {
            "groups": ["max", "presence"],
            "relation_counts": {"builds_on": 1, "inspires": 2, "references": 1},
            "dominant_relation": "inspires",
            "total_edges": 4,
            "imbalance_score": 0.5,
        },
    ]


def test_analyze_relation_imbalance_groups_by_content_type(store: Store):
    for unit in [
        _unit("unit-insight-a", "Insight A", content_type=ContentType.INSIGHT),
        _unit("unit-insight-b", "Insight B", content_type=ContentType.INSIGHT),
        _unit("unit-finding-a", "Finding A", content_type=ContentType.FINDING),
        _unit("unit-finding-b", "Finding B", content_type=ContentType.FINDING),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-1", "unit-insight-a", "unit-finding-a", EdgeRelation.CHALLENGES),
        _edge("edge-2", "unit-finding-b", "unit-insight-b", EdgeRelation.CHALLENGES),
        _edge("edge-3", "unit-insight-a", "unit-insight-b", EdgeRelation.REFINES),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_relation_imbalance(group_by="content_type")

    assert result == {
        "group_by": "content_type",
        "pair_count": 2,
        "pairs": [
            {
                "groups": ["finding", "insight"],
                "relation_counts": {"challenges": 2},
                "dominant_relation": "challenges",
                "total_edges": 2,
                "imbalance_score": 1.0,
            },
            {
                "groups": ["insight", "insight"],
                "relation_counts": {"refines": 1},
                "dominant_relation": "refines",
                "total_edges": 1,
                "imbalance_score": 1.0,
            },
        ],
    }


def test_analyze_relation_imbalance_uses_symmetric_pair_ordering(store: Store):
    for unit in [
        _unit("unit-max-a", "Max A", source_project=SourceProject.MAX),
        _unit("unit-max-b", "Max B", source_project=SourceProject.MAX),
        _unit("unit-presence-a", "Presence A", source_project=SourceProject.PRESENCE),
        _unit("unit-presence-b", "Presence B", source_project=SourceProject.PRESENCE),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-forward", "unit-max-a", "unit-presence-a", EdgeRelation.RELATES_TO),
        _edge("edge-reverse", "unit-presence-b", "unit-max-b", EdgeRelation.RELATES_TO),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_relation_imbalance()

    assert result["pairs"] == [
        {
            "groups": ["max", "presence"],
            "relation_counts": {"relates_to": 2},
            "dominant_relation": "relates_to",
            "total_edges": 2,
            "imbalance_score": 1.0,
        }
    ]


def test_analyze_relation_imbalance_skips_edges_with_missing_endpoints(store: Store):
    store.insert_unit(_unit("unit-a", "A"))
    store.insert_unit(_unit("unit-b", "B", source_project=SourceProject.PRESENCE))
    store.insert_edge(_edge("edge-valid", "unit-a", "unit-b", EdgeRelation.RELATES_TO))

    store.conn.execute("PRAGMA foreign_keys=OFF")
    store.conn.execute(
        """INSERT INTO edges
           (id, from_unit_id, to_unit_id, relation, weight, source, metadata, created_at)
           VALUES (?, ?, ?, ?, 1.0, 'manual', '{}', '2026-01-01T00:00:00+00:00')""",
        ("edge-missing", "unit-a", "unit-missing", EdgeRelation.BUILDS_ON.value),
    )
    store.conn.commit()
    store.conn.execute("PRAGMA foreign_keys=ON")

    result = GraphService(store).analyze_relation_imbalance()

    assert result["pair_count"] == 1
    assert result["pairs"][0]["relation_counts"] == {"relates_to": 1}
    assert result["pairs"][0]["total_edges"] == 1


def test_analyze_relation_imbalance_empty_graph(store: Store):
    assert GraphService(store).analyze_relation_imbalance() == {
        "group_by": "source_project",
        "pair_count": 0,
        "pairs": [],
    }


def test_analyze_relation_imbalance_validates_group_by(store: Store):
    with pytest.raises(ValueError, match="group_by must be source_project or content_type"):
        GraphService(store).analyze_relation_imbalance(group_by="tags")
