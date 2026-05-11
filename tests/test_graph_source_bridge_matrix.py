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
    source_project: SourceProject | str = SourceProject.MAX,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.INFERRED,
    )


def test_analyze_source_bridge_matrix_groups_cross_source_pairs(store: Store):
    for item in [
        _unit("a", "Alpha", source_project=SourceProject.MAX),
        _unit("b", "Beta", source_project=SourceProject.MAX),
        _unit("c", "Gamma", source_project=SourceProject.PRESENCE),
        _unit("d", "Delta", source_project=SourceProject.FORTY_TWO),
    ]:
        store.insert_unit(item)
    for item in [
        _edge("e1", "a", "c", EdgeRelation.REFERENCES),
        _edge("e2", "b", "c", EdgeRelation.REFERENCES),
        _edge("e3", "c", "d", EdgeRelation.BUILDS_ON),
        _edge("e4", "a", "b", EdgeRelation.RELATES_TO),
    ]:
        store.insert_edge(item)

    result = GraphService(store).analyze_source_bridge_matrix(limit=1)

    assert result["inter_source_edge_count"] == 3
    assert result["intra_source_edge_count"] == 1
    assert result["inter_source_ratio"] == 0.75
    assert result["relation_breakdown"] == {"builds_on": 1, "references": 2}
    assert result["source_pairs"][0] == {
        "from_source": "max",
        "to_source": "presence",
        "edge_count": 2,
        "relation_breakdown": {"references": 2},
        "edge_ids": ["e1"],
        "representative_units": [
            {
                "id": "a",
                "source_project": "max",
                "source_id": "source-a",
                "source_entity_type": "insight",
                "title": "Alpha",
                "content_type": "insight",
            }
        ],
    }


def test_analyze_source_bridge_matrix_can_include_intra_source(store: Store):
    store.insert_unit(_unit("a", "Alpha"))
    store.insert_unit(_unit("b", "Beta"))
    store.insert_edge(_edge("e1", "a", "b"))

    result = GraphService(store).analyze_source_bridge_matrix(include_intra_source=True)

    assert result["source_pairs"][0]["from_source"] == "max"
    assert result["source_pairs"][0]["to_source"] == "max"
    assert result["relation_breakdown"] == {"relates_to": 1}


@pytest.mark.parametrize("limit", [-1, "bad", True])
def test_analyze_source_bridge_matrix_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_source_bridge_matrix(limit=limit)
