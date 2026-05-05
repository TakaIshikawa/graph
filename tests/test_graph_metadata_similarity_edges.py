from __future__ import annotations

import os
import tempfile

import pytest

from graph.graph.service import GraphService, suggest_metadata_similarity_edges
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


def _unit(unit_id: str, title: str, metadata: dict) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def test_suggest_metadata_similarity_edges_handles_no_units():
    assert suggest_metadata_similarity_edges([], []) == {
        "metadata_keys": [],
        "min_overlap": 2,
        "top_n": 20,
        "unit_count": 0,
        "edge_count": 0,
        "candidate_count": 0,
        "candidates": [],
    }


def test_suggest_metadata_similarity_edges_excludes_existing_edges(store: Store):
    for unit in [
        _unit("unit-alpha", "Alpha", {"topics": ["solar", "grid"], "owner": "ops"}),
        _unit("unit-beta", "Beta", {"topics": ["solar", "grid"], "owner": "ops"}),
        _unit("unit-gamma", "Gamma", {"topics": ["solar", "grid"], "owner": "ops"}),
    ]:
        store.insert_unit(unit)
    store.insert_edge(_edge("edge-alpha-beta", "unit-alpha", "unit-beta"))

    result = GraphService(store).suggest_metadata_similarity_edges(min_overlap=3)

    assert result["candidate_count"] == 2
    assert ["unit-alpha", "unit-beta"] not in [
        candidate["unit_ids"] for candidate in result["candidates"]
    ]
    assert [candidate["unit_ids"] for candidate in result["candidates"]] == [
        ["unit-alpha", "unit-gamma"],
        ["unit-beta", "unit-gamma"],
    ]


def test_suggest_metadata_similarity_edges_compares_list_and_scalar_metadata(
    store: Store,
):
    for unit in [
        _unit(
            "unit-alpha",
            "Alpha",
            {"topics": ["solar", "storage"], "owner": "research"},
        ),
        _unit("unit-beta", "Beta", {"topics": "solar", "owner": "research"}),
        _unit("unit-gamma", "Gamma", {"topics": ["storage"], "owner": "ops"}),
    ]:
        store.insert_unit(unit)

    result = GraphService(store).suggest_metadata_similarity_edges(min_overlap=2)

    assert result["candidate_count"] == 1
    assert result["candidates"][0] == {
        "unit_ids": ["unit-alpha", "unit-beta"],
        "from_unit_id": "unit-alpha",
        "to_unit_id": "unit-beta",
        "score": 2.0,
        "overlap_count": 2,
        "shared_metadata": [
            {"key": "owner", "values": ["research"]},
            {"key": "topics", "values": ["solar"]},
        ],
        "suggested_edge": {
            "relation": "relates_to",
            "source": "inferred",
            "metadata": {
                "suggestion_reason": "metadata_similarity",
                "score": 2.0,
                "overlap_count": 2,
                "shared_metadata": [
                    {"key": "owner", "values": ["research"]},
                    {"key": "topics", "values": ["solar"]},
                ],
            },
        },
        "from_title": "Alpha",
        "to_title": "Beta",
    }


def test_suggest_metadata_similarity_edges_filters_by_metadata_keys(store: Store):
    for unit in [
        _unit("unit-alpha", "Alpha", {"topics": ["solar"], "owner": "research"}),
        _unit("unit-beta", "Beta", {"topics": ["grid"], "owner": "research"}),
        _unit("unit-gamma", "Gamma", {"topics": ["solar"], "owner": "ops"}),
    ]:
        store.insert_unit(unit)

    result = GraphService(store).suggest_metadata_similarity_edges(
        metadata_keys=["topics"],
        min_overlap=1,
    )

    assert result["metadata_keys"] == ["topics"]
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["unit_ids"] == ["unit-alpha", "unit-gamma"]
    assert result["candidates"][0]["shared_metadata"] == [
        {"key": "topics", "values": ["solar"]}
    ]


@pytest.mark.parametrize("min_overlap", [0, -1, True, "many"])
def test_suggest_metadata_similarity_edges_validates_min_overlap(
    store: Store,
    min_overlap: object,
):
    with pytest.raises(ValueError, match="min_overlap must be an integer >= 1"):
        GraphService(store).suggest_metadata_similarity_edges(
            min_overlap=min_overlap,
        )


@pytest.mark.parametrize("top_n", [-1, True, "many"])
def test_suggest_metadata_similarity_edges_validates_top_n(
    store: Store,
    top_n: object,
):
    with pytest.raises(ValueError, match="top_n must be an integer >= 0"):
        GraphService(store).suggest_metadata_similarity_edges(
            top_n=top_n,
        )


def test_suggest_metadata_similarity_edges_validates_metadata_keys(store: Store):
    with pytest.raises(ValueError, match="metadata_keys must be a sequence of strings"):
        GraphService(store).suggest_metadata_similarity_edges(metadata_keys="topics")


def test_suggest_metadata_similarity_edges_top_n_keeps_candidate_count(
    store: Store,
):
    for unit in [
        _unit("unit-alpha", "Alpha", {"topics": ["solar"], "owner": "research"}),
        _unit("unit-beta", "Beta", {"topics": ["solar"], "owner": "research"}),
        _unit("unit-gamma", "Gamma", {"topics": ["solar"], "owner": "research"}),
    ]:
        store.insert_unit(unit)

    result = GraphService(store).suggest_metadata_similarity_edges(
        min_overlap=2,
        top_n=0,
    )

    assert result["candidate_count"] == 3
    assert result["candidates"] == []


def test_suggest_metadata_similarity_edges_ranking_is_deterministic(store: Store):
    for unit in [
        _unit("unit-delta", "Delta", {"topics": ["solar"], "owner": "research"}),
        _unit("unit-alpha", "Alpha", {"topics": ["solar"], "owner": "research"}),
        _unit("unit-gamma", "Gamma", {"topics": ["solar"], "owner": "research"}),
        _unit("unit-beta", "Beta", {"topics": ["solar"], "owner": "research"}),
    ]:
        store.insert_unit(unit)

    result = GraphService(store).suggest_metadata_similarity_edges(
        min_overlap=2,
        top_n=4,
    )

    assert [candidate["unit_ids"] for candidate in result["candidates"]] == [
        ["unit-alpha", "unit-beta"],
        ["unit-alpha", "unit-delta"],
        ["unit-alpha", "unit-gamma"],
        ["unit-beta", "unit-delta"],
    ]
