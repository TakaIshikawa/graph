from __future__ import annotations

import os
import tempfile
from typing import cast

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
    tags: list[str] | None = None,
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
        tags=tags or [],
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def _insert_orphan_graph(store: Store) -> GraphService:
    for unit in [
        _unit("unit-alpha", "Alpha", tags=["energy", "solar"]),
        _unit("unit-beta", "Beta", tags=["energy", "grid"]),
        _unit("unit-gamma", "Gamma", tags=["grid", "storage"]),
        _unit(
            "unit-isolated",
            "Isolated",
            tags=["archive"],
            source_project=SourceProject.PRESENCE,
            content_type=ContentType.ARTIFACT,
        ),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-alpha-beta", "unit-alpha", "unit-beta"),
        _edge("edge-beta-gamma", "unit-beta", "unit-gamma"),
    ]:
        store.insert_edge(edge)

    service = GraphService(store)
    service.rebuild()
    return service


def test_analyze_orphan_units_reports_isolated_units_with_metadata(store: Store):
    result = _insert_orphan_graph(store).analyze_orphan_units()

    assert result == {
        "min_degree": 1,
        "node_count": 4,
        "edge_count": 2,
        "candidate_count": 1,
        "include_metadata": True,
        "units": [
            {
                "unit_id": "unit-isolated",
                "title": "Isolated",
                "degree": 0,
                "in_degree": 0,
                "out_degree": 0,
                "source_project": "presence",
                "source_id": "source-unit-isolated",
                "source_entity_type": "insight",
                "content_type": "artifact",
                "tags": ["archive"],
                "suggested_neighboring_tags": [],
            }
        ],
    }


def test_analyze_orphan_units_threshold_includes_weakly_connected_units(
    store: Store,
):
    result = _insert_orphan_graph(store).analyze_orphan_units(min_degree=2)

    assert [unit["unit_id"] for unit in result["units"]] == [
        "unit-isolated",
        "unit-alpha",
        "unit-gamma",
    ]
    assert result["candidate_count"] == 3
    assert result["units"][1] == {
        "unit_id": "unit-alpha",
        "title": "Alpha",
        "degree": 1,
        "in_degree": 0,
        "out_degree": 1,
        "source_project": "max",
        "source_id": "source-unit-alpha",
        "source_entity_type": "insight",
        "content_type": "insight",
        "tags": ["energy", "solar"],
        "suggested_neighboring_tags": [{"tag": "grid", "neighbor_count": 1}],
    }
    assert result["units"][2]["suggested_neighboring_tags"] == [
        {"tag": "energy", "neighbor_count": 1}
    ]


def test_analyze_orphan_units_can_omit_metadata(store: Store):
    result = _insert_orphan_graph(store).analyze_orphan_units(
        min_degree=2,
        include_metadata=False,
    )

    assert result["include_metadata"] is False
    assert result["units"][0] == {
        "unit_id": "unit-isolated",
        "title": "Isolated",
        "degree": 0,
        "in_degree": 0,
        "out_degree": 0,
    }
    assert "tags" not in result["units"][0]


def test_analyze_orphan_units_handles_empty_graph(store: Store):
    assert GraphService(store).analyze_orphan_units() == {
        "min_degree": 1,
        "node_count": 0,
        "edge_count": 0,
        "candidate_count": 0,
        "include_metadata": True,
        "units": [],
    }


def test_analyze_orphan_units_handles_fully_connected_graph(store: Store):
    for unit in [
        _unit("unit-a", "A"),
        _unit("unit-b", "B"),
        _unit("unit-c", "C"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-a-b", "unit-a", "unit-b"),
        _edge("edge-b-c", "unit-b", "unit-c"),
        _edge("edge-c-a", "unit-c", "unit-a"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_orphan_units()

    assert result["node_count"] == 3
    assert result["edge_count"] == 3
    assert result["candidate_count"] == 0
    assert result["units"] == []


@pytest.mark.parametrize("min_degree", [-1, True, "many"])
def test_analyze_orphan_units_validates_min_degree(
    store: Store,
    min_degree: object,
):
    with pytest.raises(ValueError, match="min_degree must be a non-negative integer"):
        GraphService(store).analyze_orphan_units(min_degree=cast(int, min_degree))
