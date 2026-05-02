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


def _insert_unit(store: Store, unit_id: str, title: str):
    return store.insert_unit(
        KnowledgeUnit(
            id=unit_id,
            source_project=SourceProject.MAX,
            source_id=unit_id,
            source_entity_type="insight",
            title=title,
            content=f"{title} content",
            content_type=ContentType.INSIGHT,
        )
    )


def _insert_edge(store: Store, from_unit_id: str, to_unit_id: str):
    return store.insert_edge(
        KnowledgeEdge(
            from_unit_id=from_unit_id,
            to_unit_id=to_unit_id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.MANUAL,
        )
    )


@pytest.fixture
def degree_store(store: Store):
    units = {
        "hub": _insert_unit(store, "hub", "Hub"),
        "inbound": _insert_unit(store, "inbound", "Inbound Heavy"),
        "outbound": _insert_unit(store, "outbound", "Outbound Heavy"),
        "leaf-a": _insert_unit(store, "leaf-a", "Leaf A"),
        "leaf-b": _insert_unit(store, "leaf-b", "Leaf B"),
        "isolated": _insert_unit(store, "isolated", "Isolated"),
    }

    for from_id, to_id in [
        ("hub", "inbound"),
        ("hub", "leaf-a"),
        ("hub", "leaf-b"),
        ("outbound", "hub"),
        ("outbound", "inbound"),
        ("outbound", "leaf-a"),
        ("outbound", "leaf-b"),
        ("leaf-b", "inbound"),
    ]:
        _insert_edge(store, units[from_id].id, units[to_id].id)

    return store


def test_degree_distribution_total_direction_summarizes_histogram_and_top_units(
    degree_store: Store,
):
    result = GraphService(degree_store).analyze_degree_distribution(top_n=3)

    assert result["direction"] == "total"
    assert result["total_units"] == 6
    assert result["isolated_unit_count"] == 1
    assert result["histogram"] == [
        {"degree": 0, "unit_count": 1},
        {"degree": 2, "unit_count": 1},
        {"degree": 3, "unit_count": 2},
        {"degree": 4, "unit_count": 2},
    ]
    assert [
        (unit["id"], unit["degree"], unit["in_degree"], unit["out_degree"])
        for unit in result["top_units"]
    ] == [
        ("hub", 4, 1, 3),
        ("outbound", 4, 0, 4),
        ("inbound", 3, 3, 0),
    ]
    assert result["top_units"][0]["title"] == "Hub"
    assert result["top_units"][0]["source_project"] == "max"
    assert result["top_units"][0]["content_type"] == "insight"


def test_degree_distribution_directions_produce_distinct_deterministic_results(
    degree_store: Store,
):
    service = GraphService(degree_store)

    total = service.analyze_degree_distribution(direction="total", top_n=2)
    inbound = service.analyze_degree_distribution(direction="in", top_n=2)
    outbound = service.analyze_degree_distribution(direction="out", top_n=2)

    assert [unit["id"] for unit in total["top_units"]] == ["hub", "outbound"]
    assert [unit["id"] for unit in inbound["top_units"]] == ["inbound", "leaf-b"]
    assert [unit["id"] for unit in outbound["top_units"]] == ["outbound", "hub"]
    assert inbound["histogram"] == [
        {"degree": 0, "unit_count": 2},
        {"degree": 1, "unit_count": 1},
        {"degree": 2, "unit_count": 2},
        {"degree": 3, "unit_count": 1},
    ]
    assert outbound["histogram"] == [
        {"degree": 0, "unit_count": 3},
        {"degree": 1, "unit_count": 1},
        {"degree": 3, "unit_count": 1},
        {"degree": 4, "unit_count": 1},
    ]


def test_degree_distribution_empty_graph_returns_empty_summary(store: Store):
    assert GraphService(store).analyze_degree_distribution() == {
        "direction": "total",
        "total_units": 0,
        "isolated_unit_count": 0,
        "histogram": [],
        "top_units": [],
    }


def test_degree_distribution_rejects_invalid_direction(store: Store):
    with pytest.raises(ValueError, match="Use 'total', 'in', or 'out'"):
        GraphService(store).analyze_degree_distribution(direction="incoming")


@pytest.mark.parametrize("top_n", [0, -1, "many", None, True])
def test_degree_distribution_rejects_invalid_top_n(store: Store, top_n):
    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        GraphService(store).analyze_degree_distribution(top_n=top_n)
