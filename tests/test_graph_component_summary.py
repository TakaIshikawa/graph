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
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
        source=EdgeSource.INFERRED,
    )


def _populate_component_graph(store: Store) -> None:
    for unit in [
        _unit("unit-alpha", "Alpha", tags=["energy", "solar"]),
        _unit(
            "unit-beta",
            "Beta",
            source_project=SourceProject.FORTY_TWO,
            tags=["energy", "grid"],
        ),
        _unit("unit-gamma", "Gamma", tags=["energy", "storage"]),
        _unit(
            "unit-delta",
            "Delta",
            source_project=SourceProject.PRESENCE,
            tags=["writing"],
        ),
        _unit(
            "unit-epsilon",
            "Epsilon",
            source_project=SourceProject.PRESENCE,
            tags=["writing", "draft"],
        ),
        _unit("unit-isolated", "Isolated", tags=["solo"]),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-alpha-beta", "unit-alpha", "unit-beta"),
        _edge("edge-beta-alpha", "unit-beta", "unit-alpha"),
        _edge("edge-beta-gamma", "unit-beta", "unit-gamma"),
        _edge("edge-gamma-alpha", "unit-gamma", "unit-alpha"),
        _edge("edge-epsilon-delta", "unit-epsilon", "unit-delta"),
    ]:
        store.insert_edge(edge)


def test_component_summary_returns_deterministic_component_metrics(store: Store):
    _populate_component_graph(store)

    result = GraphService(store).component_summary()

    assert result == {
        "node_count": 6,
        "edge_count": 5,
        "component_count": 3,
        "isolated_component_count": 1,
        "components": [
            {
                "component_id": "component-001",
                "size": 3,
                "edge_count": 4,
                "density": 0.666667,
                "representative_unit_ids": [
                    "unit-alpha",
                    "unit-beta",
                    "unit-gamma",
                ],
                "representative_titles": ["Alpha", "Beta", "Gamma"],
                "top_tags": [
                    {"tag": "energy", "count": 3},
                    {"tag": "grid", "count": 1},
                    {"tag": "solar", "count": 1},
                    {"tag": "storage", "count": 1},
                ],
                "source_project_counts": {"max": 2, "forty_two": 1},
                "isolated": False,
            },
            {
                "component_id": "component-002",
                "size": 2,
                "edge_count": 1,
                "density": 0.5,
                "representative_unit_ids": ["unit-delta", "unit-epsilon"],
                "representative_titles": ["Delta", "Epsilon"],
                "top_tags": [
                    {"tag": "writing", "count": 2},
                    {"tag": "draft", "count": 1},
                ],
                "source_project_counts": {"presence": 2},
                "isolated": False,
            },
            {
                "component_id": "component-003",
                "size": 1,
                "edge_count": 0,
                "density": 0,
                "representative_unit_ids": ["unit-isolated"],
                "representative_titles": ["Isolated"],
                "top_tags": [{"tag": "solo", "count": 1}],
                "source_project_counts": {"max": 1},
                "isolated": True,
            },
        ],
    }


def test_component_summary_applies_min_size_and_limit(store: Store):
    _populate_component_graph(store)

    result = GraphService(store).component_summary(min_size=2, limit=1)

    assert result["component_count"] == 2
    assert result["isolated_component_count"] == 0
    assert len(result["components"]) == 1
    assert result["components"][0]["component_id"] == "component-001"
    assert result["components"][0]["size"] == 3


def test_component_summary_caps_representative_lists(store: Store):
    store.insert_unit(_unit("unit-hub", "Hub"))
    for index in range(6):
        unit_id = f"unit-leaf-{index}"
        store.insert_unit(_unit(unit_id, f"Leaf {index}"))
        store.insert_edge(_edge(f"edge-{index}", "unit-hub", unit_id))

    result = GraphService(store).component_summary()

    component = result["components"][0]
    assert component["size"] == 7
    assert len(component["representative_unit_ids"]) == 5
    assert component["representative_unit_ids"][0] == "unit-hub"
    assert len(component["representative_titles"]) == 5


def test_component_summary_handles_empty_graph(store: Store):
    assert GraphService(store).component_summary() == {
        "node_count": 0,
        "edge_count": 0,
        "component_count": 0,
        "isolated_component_count": 0,
        "components": [],
    }


@pytest.mark.parametrize("min_size", [0, -1, True, "2"])
def test_component_summary_validates_min_size(store: Store, min_size: object):
    with pytest.raises(ValueError, match="min_size must be a positive integer"):
        GraphService(store).component_summary(min_size=min_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [-1, True, "many"])
def test_component_summary_validates_limit(store: Store, limit: object):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).component_summary(limit=limit)  # type: ignore[arg-type]
