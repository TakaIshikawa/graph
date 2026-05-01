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


def add_unit(store: Store, unit_id: str, title: str) -> KnowledgeUnit:
    return store.insert_unit(
        KnowledgeUnit(
            id=unit_id,
            source_project=SourceProject.MAX,
            source_id=f"source-{unit_id}",
            source_entity_type="insight",
            title=title,
            content=f"{title} content",
        )
    )


def add_edge(store: Store, from_id: str, to_id: str) -> None:
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.RELATES_TO,
        )
    )


def test_detect_communities_returns_deterministic_records(store: Store):
    for unit_id, title in [
        ("unit-a", "Alpha"),
        ("unit-b", "Beta"),
        ("unit-c", "Gamma"),
        ("unit-d", "Delta"),
        ("unit-e", "Epsilon"),
        ("unit-f", "Isolated"),
    ]:
        add_unit(store, unit_id, title)
    add_edge(store, "unit-a", "unit-b")
    add_edge(store, "unit-b", "unit-c")
    add_edge(store, "unit-d", "unit-e")

    service = GraphService(store)
    first = service.detect_communities()
    second = service.detect_communities()

    assert first == second
    assert first == [
        {
            "community_id": first[0]["community_id"],
            "size": 3,
            "unit_ids": ["unit-a", "unit-b", "unit-c"],
            "representative_titles": ["Beta", "Alpha", "Gamma"],
            "internal_edge_count": 2,
            "density": 0.666667,
        },
        {
            "community_id": first[1]["community_id"],
            "size": 2,
            "unit_ids": ["unit-d", "unit-e"],
            "representative_titles": ["Delta", "Epsilon"],
            "internal_edge_count": 1,
            "density": 1.0,
        },
    ]
    assert first[0]["community_id"].startswith("community-")
    assert first[1]["community_id"].startswith("community-")
    assert first[0]["community_id"] != first[1]["community_id"]


def test_detect_communities_honors_min_size_and_limit(store: Store):
    for unit_id, title in [
        ("unit-a", "Alpha"),
        ("unit-b", "Beta"),
        ("unit-c", "Gamma"),
        ("unit-d", "Delta"),
        ("unit-e", "Epsilon"),
        ("unit-f", "Zeta"),
    ]:
        add_unit(store, unit_id, title)
    add_edge(store, "unit-a", "unit-b")
    add_edge(store, "unit-b", "unit-c")
    add_edge(store, "unit-d", "unit-e")

    service = GraphService(store)

    assert [item["unit_ids"] for item in service.detect_communities(min_size=1)] == [
        ["unit-a", "unit-b", "unit-c"],
        ["unit-d", "unit-e"],
        ["unit-f"],
    ]
    assert service.detect_communities(min_size=3, limit=1)[0]["unit_ids"] == [
        "unit-a",
        "unit-b",
        "unit-c",
    ]


def test_detect_communities_empty_graph_returns_empty_list(store: Store):
    assert GraphService(store).detect_communities() == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_size": 0}, "min_size"),
        ({"min_size": -1}, "min_size"),
        ({"min_size": "2"}, "min_size"),
        ({"limit": 0}, "limit"),
        ({"limit": -1}, "limit"),
        ({"limit": "2"}, "limit"),
    ],
)
def test_detect_communities_validates_arguments(store: Store, kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        GraphService(store).detect_communities(**kwargs)
