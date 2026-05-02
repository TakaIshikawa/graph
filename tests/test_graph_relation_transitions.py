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


def _unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
    )


def _insert_units(store: Store, unit_ids: list[str]) -> None:
    for unit_id in unit_ids:
        store.insert_unit(_unit(unit_id, unit_id.replace("unit-", "").title()))


def _insert_edges(
    store: Store,
    edges: list[tuple[str, str, str, EdgeRelation]],
) -> None:
    for edge_id, from_unit_id, to_unit_id, relation in edges:
        store.insert_edge(_edge(edge_id, from_unit_id, to_unit_id, relation))


def test_analyze_relation_transitions_counts_and_sorts_two_hop_paths(store: Store):
    _insert_units(
        store,
        [
            "unit-a",
            "unit-b",
            "unit-c",
            "unit-d",
            "unit-e",
            "unit-f",
            "unit-g",
            "unit-h",
            "unit-i",
            "unit-j",
            "unit-k",
            "unit-l",
            "unit-m",
            "unit-n",
        ],
    )
    _insert_edges(
        store,
        [
            ("edge-a-b", "unit-a", "unit-b", EdgeRelation.DISCOVERS),
            ("edge-b-c", "unit-b", "unit-c", EdgeRelation.BUILDS_ON),
            ("edge-d-e", "unit-d", "unit-e", EdgeRelation.DISCOVERS),
            ("edge-e-f", "unit-e", "unit-f", EdgeRelation.BUILDS_ON),
            ("edge-g-h", "unit-g", "unit-h", EdgeRelation.CHALLENGES),
            ("edge-h-i", "unit-h", "unit-i", EdgeRelation.REFINES),
            ("edge-j-k", "unit-j", "unit-k", EdgeRelation.BUILDS_ON),
            ("edge-k-l", "unit-k", "unit-l", EdgeRelation.REFINES),
            ("edge-m-n", "unit-m", "unit-n", EdgeRelation.INSPIRES),
            ("edge-n-m", "unit-n", "unit-m", EdgeRelation.REFINES),
        ],
    )

    result = GraphService(store).analyze_relation_transitions(limit=1)

    assert result["path_length"] == 2
    assert result["transition_count"] == 3
    assert [
        (
            transition["from_relation"],
            transition["to_relation"],
            transition["relation_sequence"],
            transition["count"],
        )
        for transition in result["transitions"]
    ] == [
        ("discovers", "builds_on", ["discovers", "builds_on"], 2),
        ("builds_on", "refines", ["builds_on", "refines"], 1),
        ("challenges", "refines", ["challenges", "refines"], 1),
    ]

    first_transition = result["transitions"][0]
    assert len(first_transition["example_paths"]) == 1
    assert first_transition["example_paths"][0]["unit_ids"] == [
        "unit-a",
        "unit-b",
        "unit-c",
    ]
    assert first_transition["example_paths"][0]["edge_ids"] == [
        "edge-a-b",
        "edge-b-c",
    ]
    assert first_transition["example_paths"][0]["relations"] == [
        "discovers",
        "builds_on",
    ]
    assert first_transition["example_paths"][0]["units"] == [
        {
            "id": "unit-a",
            "source_project": "max",
            "source_id": "source-unit-a",
            "source_entity_type": "insight",
            "title": "A",
            "content_type": "insight",
        },
        {
            "id": "unit-b",
            "source_project": "max",
            "source_id": "source-unit-b",
            "source_entity_type": "insight",
            "title": "B",
            "content_type": "insight",
        },
        {
            "id": "unit-c",
            "source_project": "max",
            "source_id": "source-unit-c",
            "source_entity_type": "insight",
            "title": "C",
            "content_type": "insight",
        },
    ]


def test_analyze_relation_transitions_allows_zero_example_limit(store: Store):
    _insert_units(store, ["unit-a", "unit-b", "unit-c"])
    _insert_edges(
        store,
        [
            ("edge-a-b", "unit-a", "unit-b", EdgeRelation.DISCOVERS),
            ("edge-b-c", "unit-b", "unit-c", EdgeRelation.BUILDS_ON),
        ],
    )

    result = GraphService(store).analyze_relation_transitions(limit=0)

    assert result["transition_count"] == 1
    assert result["transitions"][0]["count"] == 1
    assert result["transitions"][0]["example_paths"] == []


@pytest.mark.parametrize("path_length", [0, 1, 3, "2", True])
def test_analyze_relation_transitions_validates_path_length(store: Store, path_length):
    with pytest.raises(ValueError, match="path_length currently only supports 2"):
        GraphService(store).analyze_relation_transitions(path_length=path_length)


@pytest.mark.parametrize("limit", [-1, "bad", True])
def test_analyze_relation_transitions_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_relation_transitions(limit=limit)
