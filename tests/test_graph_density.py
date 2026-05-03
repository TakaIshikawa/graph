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


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
        source=EdgeSource.INFERRED,
    )


def _populate_density_graph(store: Store) -> None:
    for unit in [
        _unit("unit-a", "Alpha"),
        _unit("unit-b", "Beta"),
        _unit("unit-c", "Gamma"),
        _unit(
            "unit-d",
            "Delta",
            source_project=SourceProject.PRESENCE,
            content_type=ContentType.ARTIFACT,
        ),
        _unit(
            "unit-e",
            "Epsilon",
            source_project=SourceProject.PRESENCE,
            content_type=ContentType.ARTIFACT,
        ),
        _unit(
            "unit-f",
            "Finding",
            source_project=SourceProject.FORTY_TWO,
            content_type=ContentType.FINDING,
        ),
    ]:
        store.insert_unit(unit)

    for index, (from_unit_id, to_unit_id) in enumerate(
        [
            ("unit-a", "unit-b"),
            ("unit-b", "unit-c"),
            ("unit-c", "unit-a"),
            ("unit-a", "unit-d"),
            ("unit-d", "unit-e"),
        ]
    ):
        store.insert_edge(_edge(f"edge-{index}", from_unit_id, to_unit_id))


def test_analyze_density_returns_top_level_density_degree_and_sparsity_metrics(
    store: Store,
):
    _populate_density_graph(store)

    result = GraphService(store).analyze_density()

    assert result == {
        "node_count": 6,
        "edge_count": 5,
        "density": 0.166667,
        "weak_component_count": 2,
        "isolated_node_count": 1,
        "average_in_degree": 0.833333,
        "average_out_degree": 0.833333,
    }


def test_analyze_density_breakdowns_use_induced_subgraphs_with_sorted_keys(
    store: Store,
):
    _populate_density_graph(store)

    result = GraphService(store).analyze_density(
        by_source_project=True,
        by_content_type=True,
    )

    assert list(result["by_source_project"]) == ["forty_two", "max", "presence"]
    assert result["by_source_project"] == {
        "forty_two": {
            "node_count": 1,
            "edge_count": 0,
            "density": 0,
            "weak_component_count": 1,
            "isolated_node_count": 1,
            "average_in_degree": 0.0,
            "average_out_degree": 0.0,
        },
        "max": {
            "node_count": 3,
            "edge_count": 3,
            "density": 0.5,
            "weak_component_count": 1,
            "isolated_node_count": 0,
            "average_in_degree": 1.0,
            "average_out_degree": 1.0,
        },
        "presence": {
            "node_count": 2,
            "edge_count": 1,
            "density": 0.5,
            "weak_component_count": 1,
            "isolated_node_count": 0,
            "average_in_degree": 0.5,
            "average_out_degree": 0.5,
        },
    }
    assert list(result["by_content_type"]) == ["artifact", "finding", "insight"]
    assert result["by_content_type"] == {
        "artifact": {
            "node_count": 2,
            "edge_count": 1,
            "density": 0.5,
            "weak_component_count": 1,
            "isolated_node_count": 0,
            "average_in_degree": 0.5,
            "average_out_degree": 0.5,
        },
        "finding": {
            "node_count": 1,
            "edge_count": 0,
            "density": 0,
            "weak_component_count": 1,
            "isolated_node_count": 1,
            "average_in_degree": 0.0,
            "average_out_degree": 0.0,
        },
        "insight": {
            "node_count": 3,
            "edge_count": 3,
            "density": 0.5,
            "weak_component_count": 1,
            "isolated_node_count": 0,
            "average_in_degree": 1.0,
            "average_out_degree": 1.0,
        },
    }


def test_analyze_density_handles_empty_graph_without_division_errors(store: Store):
    assert GraphService(store).analyze_density(
        by_source_project=True,
        by_content_type=True,
    ) == {
        "node_count": 0,
        "edge_count": 0,
        "density": 0,
        "weak_component_count": 0,
        "isolated_node_count": 0,
        "average_in_degree": 0.0,
        "average_out_degree": 0.0,
        "by_source_project": {},
        "by_content_type": {},
    }


def test_analyze_density_handles_single_node_without_division_errors(store: Store):
    store.insert_unit(_unit("unit-alpha", "Alpha"))

    assert GraphService(store).analyze_density() == {
        "node_count": 1,
        "edge_count": 0,
        "density": 0,
        "weak_component_count": 1,
        "isolated_node_count": 1,
        "average_in_degree": 0.0,
        "average_out_degree": 0.0,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"by_source_project": "yes"}, "by_source_project must be a boolean"),
        ({"by_content_type": "yes"}, "by_content_type must be a boolean"),
    ],
)
def test_analyze_density_validates_breakdown_flags(store: Store, kwargs, message):
    with pytest.raises(ValueError, match=message):
        GraphService(store).analyze_density(**kwargs)
