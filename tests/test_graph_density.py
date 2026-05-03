from __future__ import annotations

import os
import tempfile

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
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
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
        source=EdgeSource.INFERRED,
    )


def _insert_units(store: Store, *unit_ids: str) -> None:
    for unit_id in unit_ids:
        store.insert_unit(_unit(unit_id, unit_id.title()))


def test_analyze_density_handles_empty_graph_without_division_errors(store: Store):
    assert GraphService(store).analyze_density() == {
        "node_count": 0,
        "edge_count": 0,
        "directed_density": 0,
        "undirected_density": 0,
        "self_loop_count": 0,
        "reciprocal_pair_count": 0,
        "sink_count": 0,
        "source_count": 0,
        "isolated_count": 0,
    }


def test_analyze_density_handles_single_node_without_division_errors(store: Store):
    _insert_units(store, "unit-a")

    assert GraphService(store).analyze_density() == {
        "node_count": 1,
        "edge_count": 0,
        "directed_density": 0,
        "undirected_density": 0,
        "self_loop_count": 0,
        "reciprocal_pair_count": 0,
        "sink_count": 1,
        "source_count": 1,
        "isolated_count": 1,
    }


def test_analyze_density_counts_directed_edges_and_densities(store: Store):
    _insert_units(store, "unit-a", "unit-b", "unit-c")
    store.insert_edge(_edge("edge-ab", "unit-a", "unit-b"))
    store.insert_edge(_edge("edge-bc", "unit-b", "unit-c"))

    assert GraphService(store).analyze_density() == {
        "node_count": 3,
        "edge_count": 2,
        "directed_density": 0.333333,
        "undirected_density": 0.666667,
        "self_loop_count": 0,
        "reciprocal_pair_count": 0,
        "sink_count": 1,
        "source_count": 1,
        "isolated_count": 0,
    }


def test_analyze_density_counts_reciprocal_pairs_once(store: Store):
    _insert_units(store, "unit-a", "unit-b")
    store.insert_edge(_edge("edge-ab", "unit-a", "unit-b"))
    store.insert_edge(_edge("edge-ba", "unit-b", "unit-a"))

    assert GraphService(store).analyze_density() == {
        "node_count": 2,
        "edge_count": 2,
        "directed_density": 1.0,
        "undirected_density": 1.0,
        "self_loop_count": 0,
        "reciprocal_pair_count": 1,
        "sink_count": 0,
        "source_count": 0,
        "isolated_count": 0,
    }


def test_analyze_density_counts_self_loops_without_inflating_density(store: Store):
    _insert_units(store, "unit-a", "unit-b")
    store.insert_edge(_edge("edge-aa", "unit-a", "unit-a"))
    store.insert_edge(_edge("edge-ab", "unit-a", "unit-b"))

    assert GraphService(store).analyze_density() == {
        "node_count": 2,
        "edge_count": 2,
        "directed_density": 0.5,
        "undirected_density": 1.0,
        "self_loop_count": 1,
        "reciprocal_pair_count": 0,
        "sink_count": 1,
        "source_count": 0,
        "isolated_count": 0,
    }


def test_analyze_density_counts_sources_sinks_and_isolates(store: Store):
    _insert_units(store, "unit-a", "unit-b", "unit-c", "unit-d")
    store.insert_edge(_edge("edge-ab", "unit-a", "unit-b"))
    store.insert_edge(_edge("edge-ac", "unit-a", "unit-c"))

    assert GraphService(store).analyze_density() == {
        "node_count": 4,
        "edge_count": 2,
        "directed_density": 0.166667,
        "undirected_density": 0.333333,
        "self_loop_count": 0,
        "reciprocal_pair_count": 0,
        "sink_count": 3,
        "source_count": 2,
        "isolated_count": 1,
    }
