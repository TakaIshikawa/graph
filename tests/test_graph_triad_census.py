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


def _unit(unit_id: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="insight",
        title=f"Unit {unit_id}",
        content=f"Content for {unit_id}",
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


def _insert_units(store: Store, *unit_ids: str) -> None:
    for unit_id in unit_ids:
        store.insert_unit(_unit(unit_id))


def _insert_edges(store: Store, *pairs: tuple[str, str]) -> None:
    for index, (from_unit_id, to_unit_id) in enumerate(pairs, start=1):
        store.insert_edge(_edge(f"edge-{index}", from_unit_id, to_unit_id))


def _zero_census() -> dict[str, int]:
    return {
        "003": 0,
        "012": 0,
        "102": 0,
        "021D": 0,
        "021U": 0,
        "021C": 0,
        "111D": 0,
        "111U": 0,
        "030T": 0,
        "030C": 0,
        "201": 0,
        "120D": 0,
        "120U": 0,
        "120C": 0,
        "210": 0,
        "300": 0,
    }


def test_analyze_triad_census_empty_graph_returns_zero_summary(store: Store):
    assert GraphService(store).analyze_triad_census() == {
        "total_triads": 0,
        "census": _zero_census(),
        "non_empty_triads": 0,
        "top_types": [],
        "node_count": 0,
    }


def test_analyze_triad_census_two_node_graph_returns_zero_summary(store: Store):
    _insert_units(store, "unit-a", "unit-b")
    _insert_edges(store, ("unit-a", "unit-b"))

    assert GraphService(store).analyze_triad_census() == {
        "total_triads": 0,
        "census": _zero_census(),
        "non_empty_triads": 0,
        "top_types": [],
        "node_count": 2,
    }


def test_analyze_triad_census_counts_directed_chain_motif(store: Store):
    _insert_units(store, "unit-a", "unit-b", "unit-c")
    _insert_edges(store, ("unit-a", "unit-b"), ("unit-b", "unit-c"))

    result = GraphService(store).analyze_triad_census()

    expected_census = _zero_census()
    expected_census["021C"] = 1
    assert result == {
        "total_triads": 1,
        "census": expected_census,
        "non_empty_triads": 1,
        "top_types": [{"type": "021C", "count": 1}],
        "node_count": 3,
    }


def test_analyze_triad_census_counts_reciprocal_motif(store: Store):
    _insert_units(store, "unit-a", "unit-b", "unit-c")
    _insert_edges(
        store,
        ("unit-a", "unit-b"),
        ("unit-b", "unit-a"),
        ("unit-b", "unit-c"),
    )

    result = GraphService(store).analyze_triad_census()

    expected_census = _zero_census()
    expected_census["111U"] = 1
    assert result["total_triads"] == 1
    assert result["census"] == expected_census
    assert result["non_empty_triads"] == 1
    assert result["top_types"] == [{"type": "111U", "count": 1}]
    assert result["node_count"] == 3


def test_analyze_triad_census_top_types_excludes_zeroes_and_sorts_ties(store: Store):
    _insert_units(store, "unit-a", "unit-b", "unit-c", "unit-d")
    _insert_edges(store, ("unit-a", "unit-b"))

    result = GraphService(store).analyze_triad_census()

    assert result["total_triads"] == 4
    assert result["census"]["012"] == 2
    assert result["census"]["003"] == 2
    assert result["non_empty_triads"] == 2
    assert result["top_types"] == [
        {"type": "003", "count": 2},
        {"type": "012", "count": 2},
    ]
