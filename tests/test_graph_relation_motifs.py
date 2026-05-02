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


def test_analyze_relation_motifs_counts_sequences_and_examples(store: Store):
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
            ("edge-a-b", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
            ("edge-b-c", "unit-b", "unit-c", EdgeRelation.CHALLENGES),
            ("edge-d-e", "unit-d", "unit-e", EdgeRelation.BUILDS_ON),
            ("edge-e-f", "unit-e", "unit-f", EdgeRelation.CHALLENGES),
            ("edge-g-h", "unit-g", "unit-h", EdgeRelation.BUILDS_ON),
            ("edge-h-i", "unit-h", "unit-i", EdgeRelation.REFINES),
            ("edge-j-k", "unit-j", "unit-k", EdgeRelation.DISCOVERS),
            ("edge-k-l", "unit-k", "unit-l", EdgeRelation.INSPIRES),
            ("edge-m-n", "unit-m", "unit-n", EdgeRelation.INSPIRES),
            ("edge-n-m", "unit-n", "unit-m", EdgeRelation.REFINES),
        ],
    )

    result = GraphService(store).analyze_relation_motifs()

    assert result["path_length"] == 2
    assert result["limit"] == 20
    assert result["min_count"] == 1
    assert result["motif_count"] == 3
    assert result["stats"] == {
        "total_paths": 4,
        "unique_motifs": 3,
        "matching_motifs": 3,
        "returned_motifs": 3,
    }
    assert [
        (motif["relation_sequence"], motif["count"])
        for motif in result["motifs"]
    ] == [
        (["builds_on", "challenges"], 2),
        (["builds_on", "refines"], 1),
        (["discovers", "inspires"], 1),
    ]

    first_motif = result["motifs"][0]
    assert first_motif["examples"][0]["unit_ids"] == [
        "unit-a",
        "unit-b",
        "unit-c",
    ]
    assert first_motif["examples"][0]["edge_ids"] == [
        "edge-a-b",
        "edge-b-c",
    ]
    assert first_motif["examples"][0]["relations"] == [
        "builds_on",
        "challenges",
    ]
    assert first_motif["examples"][0]["units"] == [
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
    assert [
        example["edge_ids"] for example in first_motif["examples"]
    ] == [
        ["edge-a-b", "edge-b-c"],
        ["edge-d-e", "edge-e-f"],
    ]
    assert first_motif["examples"][0]["edges"][0]["relation"] == "builds_on"


def test_analyze_relation_motifs_enforces_min_count_and_limit(store: Store):
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
        ],
    )
    _insert_edges(
        store,
        [
            ("edge-a-b", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
            ("edge-b-c", "unit-b", "unit-c", EdgeRelation.CHALLENGES),
            ("edge-d-e", "unit-d", "unit-e", EdgeRelation.BUILDS_ON),
            ("edge-e-f", "unit-e", "unit-f", EdgeRelation.CHALLENGES),
            ("edge-g-h", "unit-g", "unit-h", EdgeRelation.BUILDS_ON),
            ("edge-h-i", "unit-h", "unit-i", EdgeRelation.REFINES),
        ],
    )

    result = GraphService(store).analyze_relation_motifs(min_count=2, limit=1)

    assert result["motif_count"] == 1
    assert result["stats"] == {
        "total_paths": 3,
        "unique_motifs": 2,
        "matching_motifs": 1,
        "returned_motifs": 1,
    }
    assert result["motifs"][0]["relation_sequence"] == ["builds_on", "challenges"]
    assert result["motifs"][0]["count"] == 2


def test_analyze_relation_motifs_allows_zero_limit(store: Store):
    _insert_units(store, ["unit-a", "unit-b", "unit-c"])
    _insert_edges(
        store,
        [
            ("edge-a-b", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
            ("edge-b-c", "unit-b", "unit-c", EdgeRelation.CHALLENGES),
        ],
    )

    result = GraphService(store).analyze_relation_motifs(limit=0)

    assert result["motif_count"] == 0
    assert result["motifs"] == []
    assert result["stats"] == {
        "total_paths": 1,
        "unique_motifs": 1,
        "matching_motifs": 1,
        "returned_motifs": 0,
    }


@pytest.mark.parametrize("path_length", [0, 1, 3, "2", True])
def test_analyze_relation_motifs_validates_path_length(store: Store, path_length):
    with pytest.raises(ValueError, match="path_length currently only supports 2"):
        GraphService(store).analyze_relation_motifs(path_length=path_length)


@pytest.mark.parametrize("limit", [-1, "bad", True])
def test_analyze_relation_motifs_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).analyze_relation_motifs(limit=limit)


@pytest.mark.parametrize("min_count", [0, -1, 1.5, True, "2"])
def test_analyze_relation_motifs_validates_min_count(store: Store, min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        GraphService(store).analyze_relation_motifs(min_count=min_count)
