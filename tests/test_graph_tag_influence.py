from __future__ import annotations

import os
import tempfile

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _unit(unit_id: str, title: str, tags: list[str] | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        tags=tags or [],
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def test_tag_influence_ranks_connected_and_disconnected_tags(store: Store):
    for unit in [
        _unit("unit-a", "Alpha", ["alpha"]),
        _unit("unit-b", "Beta", ["alpha", "beta"]),
        _unit("unit-c", "Gamma", ["beta"]),
        _unit("unit-d", "Delta", ["gamma"]),
        _unit("unit-e", "Epsilon", ["gamma"]),
        _unit("unit-f", "Untagged"),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-a-b", "unit-a", "unit-b"),
        _edge("edge-b-c", "unit-b", "unit-c"),
        _edge("edge-b-f", "unit-b", "unit-f"),
        _edge("edge-d-e", "unit-d", "unit-e"),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).tag_influence(top_n=10)

    assert result["top_n"] == 10
    assert result["tag_count"] == 3
    assert result["stats"] == {
        "unit_count": 6,
        "edge_count": 4,
        "tagged_unit_count": 5,
        "untagged_unit_count": 1,
        "unique_tags": 3,
        "returned_tags": 3,
    }
    assert [
        (
            tag["tag"],
            tag["unit_count"],
            tag["edge_touch_count"],
            tag["average_degree"],
            tag["representative_unit_ids"],
        )
        for tag in result["tags"]
    ] == [
        ("alpha", 2, 4, 2.0, ["unit-b", "unit-a"]),
        ("beta", 2, 4, 2.0, ["unit-b", "unit-c"]),
        ("gamma", 2, 2, 1.0, ["unit-d", "unit-e"]),
    ]


def test_tag_influence_limits_results_with_stable_tie_breaking(store: Store):
    for unit in [
        _unit("unit-a", "Alpha", ["alpha"]),
        _unit("unit-b", "Beta", ["beta"]),
        _unit("unit-c", "Gamma", ["gamma"]),
    ]:
        store.insert_unit(unit)

    result = GraphService(store).tag_influence(top_n=2)

    assert [tag["tag"] for tag in result["tags"]] == ["alpha", "beta"]
    assert result["tag_count"] == 2
    assert result["stats"]["unique_tags"] == 3
    assert result["stats"]["returned_tags"] == 2


def test_tag_influence_allows_empty_and_zero_limit(store: Store):
    result = GraphService(store).tag_influence(top_n=0)

    assert result == {
        "top_n": 0,
        "tag_count": 0,
        "tags": [],
        "stats": {
            "unit_count": 0,
            "edge_count": 0,
            "tagged_unit_count": 0,
            "untagged_unit_count": 0,
            "unique_tags": 0,
            "returned_tags": 0,
        },
    }


def test_tag_influence_handles_untagged_graph(store: Store):
    store.insert_unit(_unit("unit-a", "Alpha"))
    store.insert_unit(_unit("unit-b", "Beta"))
    store.insert_edge(_edge("edge-a-b", "unit-a", "unit-b"))

    result = GraphService(store).tag_influence()

    assert result["tags"] == []
    assert result["stats"] == {
        "unit_count": 2,
        "edge_count": 1,
        "tagged_unit_count": 0,
        "untagged_unit_count": 2,
        "unique_tags": 0,
        "returned_tags": 0,
    }


@pytest.mark.parametrize("top_n", [-1, "2", 1.5, True, None])
def test_tag_influence_validates_top_n(store: Store, top_n):
    with pytest.raises(ValueError, match="top_n must be a non-negative integer"):
        GraphService(store).tag_influence(top_n=top_n)
