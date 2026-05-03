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


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def test_weak_component_summary_handles_disconnected_directed_graph(store: Store):
    for unit in [
        _unit("unit-alpha", "Alpha", tags=["energy", "solar"]),
        _unit(
            "unit-beta",
            "Beta",
            source_project=SourceProject.FORTY_TWO,
            tags=["energy", "grid"],
        ),
        _unit("unit-gamma", "Gamma", tags=["storage"]),
        _unit("unit-delta", "Delta", source_project=SourceProject.PRESENCE),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-beta-alpha", "unit-beta", "unit-alpha"),
        _edge("edge-gamma-delta", "unit-gamma", "unit-delta"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).weak_component_summary()

    assert result["node_count"] == 4
    assert result["edge_count"] == 2
    assert result["component_count"] == 2
    assert result["isolated_component_count"] == 0
    assert result["components"] == [
        {
            "component_id": "component-001",
            "unit_ids": ["unit-alpha", "unit-beta"],
            "size": 2,
            "representative_unit_ids": ["unit-alpha", "unit-beta"],
            "representative_titles": ["Alpha", "Beta"],
            "source_project_counts": {"max": 1, "forty_two": 1},
            "source_breakdown": {"max": 1, "forty_two": 1},
            "tag_breakdown": {"energy": 2, "grid": 1, "solar": 1},
            "tag_counts": {"energy": 2, "grid": 1, "solar": 1},
            "internal_edge_count": 1,
            "isolated": False,
        },
        {
            "component_id": "component-002",
            "unit_ids": ["unit-delta", "unit-gamma"],
            "size": 2,
            "representative_unit_ids": ["unit-delta", "unit-gamma"],
            "representative_titles": ["Delta", "Gamma"],
            "source_project_counts": {"presence": 1, "max": 1},
            "source_breakdown": {"presence": 1, "max": 1},
            "tag_breakdown": {"storage": 1},
            "tag_counts": {"storage": 1},
            "internal_edge_count": 1,
            "isolated": False,
        },
    ]


def test_weak_component_summary_represents_isolated_units(store: Store):
    store.insert_unit(_unit("unit-isolated", "Isolated", tags=["solo"]))

    result = GraphService(store).weak_component_summary()

    assert result == {
        "node_count": 1,
        "edge_count": 0,
        "component_count": 1,
        "isolated_component_count": 1,
        "components": [
            {
                "component_id": "component-001",
                "unit_ids": ["unit-isolated"],
                "size": 1,
                "representative_unit_ids": ["unit-isolated"],
                "representative_titles": ["Isolated"],
                "source_project_counts": {"max": 1},
                "source_breakdown": {"max": 1},
                "tag_breakdown": {"solo": 1},
                "tag_counts": {"solo": 1},
                "internal_edge_count": 0,
                "isolated": True,
            }
        ],
    }


def test_weak_component_summary_orders_ties_deterministically(store: Store):
    for unit in [
        _unit("unit-zeta", "Zeta"),
        _unit("unit-alpha", "Alpha"),
        _unit("unit-beta", "Beta"),
        _unit("unit-delta", "Delta"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-alpha-zeta", "unit-alpha", "unit-zeta"),
        _edge("edge-beta-delta", "unit-beta", "unit-delta"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_weak_components(
        limit=1,
        representative_limit=1,
    )

    assert result["component_count"] == 2
    assert len(result["components"]) == 1
    assert result["components"][0]["component_id"] == "component-001"
    assert result["components"][0]["unit_ids"] == ["unit-alpha", "unit-zeta"]
    assert result["components"][0]["representative_unit_ids"] == ["unit-alpha"]
    assert result["components"][0]["representative_titles"] == ["Alpha"]


@pytest.mark.parametrize("limit", [-1, True, "many"])
def test_weak_component_summary_validates_limit(store: Store, limit: object):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        GraphService(store).weak_component_summary(limit=limit)  # type: ignore[arg-type]


def test_analyze_weakly_connected_components_treats_directed_edges_as_undirected(
    store: Store,
):
    for unit in [
        _unit("unit-alpha", "Alpha", tags=["energy"]),
        _unit(
            "unit-beta",
            "Beta",
            source_project=SourceProject.FORTY_TWO,
            content_type=ContentType.FINDING,
        ),
        _unit("unit-gamma", "Gamma", content_type=ContentType.ARTIFACT),
        _unit("unit-delta", "Delta", source_project=SourceProject.PRESENCE),
        _unit("unit-isolated", "Isolated"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-beta-alpha", "unit-beta", "unit-alpha"),
        _edge("edge-gamma-delta", "unit-gamma", "unit-delta"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_weakly_connected_components()

    assert result["node_count"] == 5
    assert result["edge_count"] == 2
    assert result["component_count"] == 3
    assert result["largest_component_size"] == 2
    assert result["isolated_component_count"] == 1
    assert result["components"] == [
        {
            "component_id": "component-001",
            "size": 2,
            "unit_ids": ["unit-alpha", "unit-beta"],
            "source_project_counts": {"forty_two": 1, "max": 1},
            "content_type_counts": {"finding": 1, "insight": 1},
            "edge_count": 1,
            "representative_units": [
                {
                    "id": "unit-alpha",
                    "source_project": "max",
                    "source_id": "source-unit-alpha",
                    "source_entity_type": "insight",
                    "title": "Alpha",
                    "content_type": "insight",
                },
                {
                    "id": "unit-beta",
                    "source_project": "forty_two",
                    "source_id": "source-unit-beta",
                    "source_entity_type": "insight",
                    "title": "Beta",
                    "content_type": "finding",
                },
            ],
            "isolated": False,
        },
        {
            "component_id": "component-002",
            "size": 2,
            "unit_ids": ["unit-delta", "unit-gamma"],
            "source_project_counts": {"max": 1, "presence": 1},
            "content_type_counts": {"artifact": 1, "insight": 1},
            "edge_count": 1,
            "representative_units": [
                {
                    "id": "unit-delta",
                    "source_project": "presence",
                    "source_id": "source-unit-delta",
                    "source_entity_type": "insight",
                    "title": "Delta",
                    "content_type": "insight",
                },
                {
                    "id": "unit-gamma",
                    "source_project": "max",
                    "source_id": "source-unit-gamma",
                    "source_entity_type": "insight",
                    "title": "Gamma",
                    "content_type": "artifact",
                },
            ],
            "isolated": False,
        },
        {
            "component_id": "component-003",
            "size": 1,
            "unit_ids": ["unit-isolated"],
            "source_project_counts": {"max": 1},
            "content_type_counts": {"insight": 1},
            "edge_count": 0,
            "representative_units": [
                {
                    "id": "unit-isolated",
                    "source_project": "max",
                    "source_id": "source-unit-isolated",
                    "source_entity_type": "insight",
                    "title": "Isolated",
                    "content_type": "insight",
                }
            ],
            "isolated": True,
        },
    ]


def test_analyze_weakly_connected_components_limits_after_full_counts(store: Store):
    for unit in [
        _unit("unit-zeta", "Zeta"),
        _unit("unit-alpha", "Alpha"),
        _unit("unit-beta", "Beta"),
        _unit("unit-delta", "Delta"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-alpha-zeta", "unit-alpha", "unit-zeta"),
        _edge("edge-beta-delta", "unit-beta", "unit-delta"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_weakly_connected_components(limit=1)

    assert result["component_count"] == 2
    assert result["largest_component_size"] == 2
    assert len(result["components"]) == 1
    assert result["components"][0]["component_id"] == "component-001"
    assert result["components"][0]["unit_ids"] == ["unit-alpha", "unit-zeta"]


@pytest.mark.parametrize("limit", [0, -1, True, "many", None])
def test_analyze_weakly_connected_components_validates_limit(
    store: Store, limit: object
):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        GraphService(store).analyze_weakly_connected_components(
            limit=limit  # type: ignore[arg-type]
        )
