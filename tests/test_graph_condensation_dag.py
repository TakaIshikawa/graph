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
        source=EdgeSource.INFERRED,
    )


def _populate(store: Store, units: list[tuple[str, str]], edges: list[tuple[str, str]]) -> None:
    for unit_id, title in units:
        store.insert_unit(_unit(unit_id, title))
    for index, (from_unit_id, to_unit_id) in enumerate(edges):
        store.insert_edge(_edge(f"edge-{index}", from_unit_id, to_unit_id))


def test_analyze_condensation_dag_reports_acyclic_units_in_topological_order(
    store: Store,
):
    _populate(
        store,
        [
            ("unit-a", "Alpha"),
            ("unit-b", "Beta"),
            ("unit-c", "Gamma"),
            ("unit-d", "Delta"),
        ],
        [
            ("unit-a", "unit-b"),
            ("unit-a", "unit-c"),
            ("unit-b", "unit-d"),
            ("unit-c", "unit-d"),
        ],
    )

    result = GraphService(store).analyze_condensation_dag()

    assert result == {
        "component_count": 4,
        "cyclic_component_count": 0,
        "source_component_count": 1,
        "sink_component_count": 1,
        "topological_order": [
            "component-001",
            "component-002",
            "component-003",
            "component-004",
        ],
        "components": [
            {
                "component_id": "component-001",
                "size": 1,
                "unit_ids": ["unit-a"],
                "representative_titles": ["Alpha"],
                "incoming_component_count": 0,
                "outgoing_component_count": 2,
                "cyclic": False,
            },
            {
                "component_id": "component-002",
                "size": 1,
                "unit_ids": ["unit-b"],
                "representative_titles": ["Beta"],
                "incoming_component_count": 1,
                "outgoing_component_count": 1,
                "cyclic": False,
            },
            {
                "component_id": "component-003",
                "size": 1,
                "unit_ids": ["unit-c"],
                "representative_titles": ["Gamma"],
                "incoming_component_count": 1,
                "outgoing_component_count": 1,
                "cyclic": False,
            },
            {
                "component_id": "component-004",
                "size": 1,
                "unit_ids": ["unit-d"],
                "representative_titles": ["Delta"],
                "incoming_component_count": 2,
                "outgoing_component_count": 0,
                "cyclic": False,
            },
        ],
    }


def test_analyze_condensation_dag_collapses_cyclic_components(store: Store):
    _populate(
        store,
        [
            ("unit-a", "Alpha"),
            ("unit-b", "Beta"),
            ("unit-c", "Gamma"),
            ("unit-d", "Delta"),
        ],
        [
            ("unit-a", "unit-b"),
            ("unit-b", "unit-c"),
            ("unit-c", "unit-a"),
            ("unit-c", "unit-d"),
        ],
    )

    result = GraphService(store).analyze_condensation_dag()

    assert result["component_count"] == 2
    assert result["cyclic_component_count"] == 1
    assert result["source_component_count"] == 1
    assert result["sink_component_count"] == 1
    assert result["components"] == [
        {
            "component_id": "component-001",
            "size": 3,
            "unit_ids": ["unit-a", "unit-b", "unit-c"],
            "representative_titles": ["Alpha", "Beta", "Gamma"],
            "incoming_component_count": 0,
            "outgoing_component_count": 1,
            "cyclic": True,
        },
        {
            "component_id": "component-002",
            "size": 1,
            "unit_ids": ["unit-d"],
            "representative_titles": ["Delta"],
            "incoming_component_count": 1,
            "outgoing_component_count": 0,
            "cyclic": False,
        },
    ]


def test_analyze_condensation_dag_includes_isolated_nodes_as_source_sinks(
    store: Store,
):
    _populate(
        store,
        [
            ("unit-a", "Alpha"),
            ("unit-b", "Beta"),
            ("unit-c", "Gamma"),
        ],
        [("unit-a", "unit-b")],
    )

    result = GraphService(store).analyze_condensation_dag()

    assert result["component_count"] == 3
    assert result["source_component_count"] == 2
    assert result["sink_component_count"] == 2
    assert [component["unit_ids"] for component in result["components"]] == [
        ["unit-a"],
        ["unit-b"],
        ["unit-c"],
    ]
    assert result["components"][2] == {
        "component_id": "component-003",
        "size": 1,
        "unit_ids": ["unit-c"],
        "representative_titles": ["Gamma"],
        "incoming_component_count": 0,
        "outgoing_component_count": 0,
        "cyclic": False,
    }


def test_analyze_condensation_dag_applies_limit_to_component_summaries(store: Store):
    _populate(
        store,
        [
            ("unit-a", "Alpha"),
            ("unit-b", "Beta"),
            ("unit-c", "Gamma"),
        ],
        [("unit-a", "unit-b"), ("unit-b", "unit-c")],
    )

    result = GraphService(store).analyze_condensation_dag(limit=2)

    assert result["component_count"] == 3
    assert result["topological_order"] == [
        "component-001",
        "component-002",
        "component-003",
    ]
    assert [component["unit_ids"] for component in result["components"]] == [
        ["unit-a"],
        ["unit-b"],
    ]


@pytest.mark.parametrize("limit", [0, -1, "bad", True])
def test_analyze_condensation_dag_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        GraphService(store).analyze_condensation_dag(limit=limit)
