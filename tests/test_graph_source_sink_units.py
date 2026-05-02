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


def _unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def test_analyze_source_sink_units_empty_graph(store: Store):
    assert GraphService(store).analyze_source_sink_units() == {
        "sources": [],
        "sinks": [],
        "summary": {
            "total_node_count": 0,
            "source_count": 0,
            "sink_count": 0,
            "isolated_count": 0,
        },
    }


def test_analyze_source_sink_units_classifies_directional_dead_ends(store: Store):
    for unit in [
        _unit("unit-source-alpha", "Alpha source"),
        _unit("unit-source-beta", "Beta source"),
        _unit("unit-sink-alpha", "Alpha sink"),
        _unit("unit-sink-beta", "Beta sink"),
        _unit("unit-isolated", "Isolated"),
        _unit("unit-bridge-a", "Bridge A"),
        _unit("unit-bridge-b", "Bridge B"),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-1", "unit-source-alpha", "unit-sink-alpha"),
        _edge("edge-2", "unit-source-alpha", "unit-sink-beta"),
        _edge("edge-3", "unit-source-beta", "unit-sink-alpha"),
        _edge("edge-4", "unit-bridge-a", "unit-bridge-b"),
        _edge("edge-5", "unit-bridge-b", "unit-bridge-a"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_source_sink_units()

    assert result["summary"] == {
        "total_node_count": 7,
        "source_count": 2,
        "sink_count": 2,
        "isolated_count": 1,
    }
    assert result["sources"] == [
        {
            "unit": {
                "id": "unit-source-alpha",
                "source_project": "max",
                "source_id": "source-unit-source-alpha",
                "source_entity_type": "insight",
                "title": "Alpha source",
                "content_type": "insight",
            },
            "in_degree": 0,
            "out_degree": 2,
        },
        {
            "unit": {
                "id": "unit-source-beta",
                "source_project": "max",
                "source_id": "source-unit-source-beta",
                "source_entity_type": "insight",
                "title": "Beta source",
                "content_type": "insight",
            },
            "in_degree": 0,
            "out_degree": 1,
        },
    ]
    assert result["sinks"] == [
        {
            "unit": {
                "id": "unit-sink-alpha",
                "source_project": "max",
                "source_id": "source-unit-sink-alpha",
                "source_entity_type": "insight",
                "title": "Alpha sink",
                "content_type": "insight",
            },
            "in_degree": 2,
            "out_degree": 0,
        },
        {
            "unit": {
                "id": "unit-sink-beta",
                "source_project": "max",
                "source_id": "source-unit-sink-beta",
                "source_entity_type": "insight",
                "title": "Beta sink",
                "content_type": "insight",
            },
            "in_degree": 1,
            "out_degree": 0,
        },
    ]


def test_analyze_source_sink_units_limit_is_deterministic(store: Store):
    for unit in [
        _unit("unit-source-alpha", "Alpha source"),
        _unit("unit-source-beta", "Beta source"),
        _unit("unit-sink-alpha", "Alpha sink"),
        _unit("unit-sink-beta", "Beta sink"),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-1", "unit-source-alpha", "unit-sink-alpha"),
        _edge("edge-2", "unit-source-beta", "unit-sink-beta"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_source_sink_units(limit=1)

    assert result["summary"]["source_count"] == 2
    assert result["summary"]["sink_count"] == 2
    assert [item["unit"]["id"] for item in result["sources"]] == ["unit-source-alpha"]
    assert [item["unit"]["id"] for item in result["sinks"]] == ["unit-sink-alpha"]


def test_analyze_source_sink_units_validates_limit(store: Store):
    service = GraphService(store)

    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        service.analyze_source_sink_units(limit=-1)

    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        service.analyze_source_sink_units(limit=True)

