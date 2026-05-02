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
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"Content for {title}",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
        source=EdgeSource.MANUAL,
    )


@pytest.fixture
def orphan_topic_store(store: Store):
    for unit in [
        _unit("unit-connected-a", "Connected A", tags=["systems"]),
        _unit("unit-connected-b", "Connected B", tags=["systems"]),
        _unit("unit-connected-c", "Connected C", tags=["systems"]),
        _unit(
            "unit-isolated-tagged",
            "Tagged Isolate",
            source_project=SourceProject.PRESENCE,
            tags=["archive", "topic"],
        ),
        _unit("unit-isolated-plain", "Plain Isolate"),
        _unit("unit-leaf", "Sparse Tagged Leaf", tags=["research"]),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-a-b", "unit-connected-a", "unit-connected-b"),
        _edge("edge-b-c", "unit-connected-b", "unit-connected-c"),
        _edge("edge-c-a", "unit-connected-c", "unit-connected-a"),
        _edge("edge-leaf-a", "unit-leaf", "unit-connected-a"),
    ]:
        store.insert_edge(edge)

    return store


def test_analyze_orphan_topics_returns_context_and_deterministic_order(
    orphan_topic_store: Store,
):
    result = GraphService(orphan_topic_store).analyze_orphan_topics(min_degree=2)

    assert result["summary"] == {
        "total_units": 6,
        "candidate_count": 3,
        "returned_count": 3,
        "min_degree": 2,
        "include_isolated_tag_only_units": True,
        "limit": 50,
    }
    assert [topic["unit_id"] for topic in result["topics"]] == [
        "unit-isolated-plain",
        "unit-isolated-tagged",
        "unit-leaf",
    ]
    assert result["topics"][0] == {
        "unit_id": "unit-isolated-plain",
        "title": "Plain Isolate",
        "source_project": "max",
        "source_id": "source-unit-isolated-plain",
        "source_entity_type": "insight",
        "content_type": "insight",
        "tags": [],
        "degree": 0,
        "in_degree": 0,
        "out_degree": 0,
        "reason_code": "isolated",
    }
    assert result["topics"][1]["tags"] == ["archive", "topic"]
    assert result["topics"][1]["reason_code"] == "tag_only_isolate"
    assert result["topics"][2]["degree"] == 1
    assert result["topics"][2]["out_degree"] == 1
    assert result["topics"][2]["reason_code"] == "low_degree"


def test_analyze_orphan_topics_threshold_controls_low_degree_inclusion(
    orphan_topic_store: Store,
):
    result = GraphService(orphan_topic_store).analyze_orphan_topics(min_degree=1)

    assert [topic["unit_id"] for topic in result["topics"]] == [
        "unit-isolated-plain",
        "unit-isolated-tagged",
    ]
    assert {topic["reason_code"] for topic in result["topics"]} == {
        "isolated",
        "tag_only_isolate",
    }


def test_analyze_orphan_topics_can_exclude_isolated_tag_only_units(
    orphan_topic_store: Store,
):
    result = GraphService(orphan_topic_store).analyze_orphan_topics(
        min_degree=2,
        include_isolated_tag_only_units=False,
    )

    assert [topic["unit_id"] for topic in result["topics"]] == [
        "unit-isolated-plain",
        "unit-leaf",
    ]
    assert result["summary"]["candidate_count"] == 2
    assert result["summary"]["include_isolated_tag_only_units"] is False


def test_analyze_orphan_topics_applies_limit_after_deterministic_sort(
    orphan_topic_store: Store,
):
    result = GraphService(orphan_topic_store).analyze_orphan_topics(min_degree=2, limit=2)

    assert result["summary"]["candidate_count"] == 3
    assert result["summary"]["returned_count"] == 2
    assert [topic["unit_id"] for topic in result["topics"]] == [
        "unit-isolated-plain",
        "unit-isolated-tagged",
    ]


@pytest.mark.parametrize("min_degree", [-1, "many", True])
def test_analyze_orphan_topics_validates_min_degree(store: Store, min_degree):
    with pytest.raises(ValueError, match="min_degree must be a non-negative integer"):
        GraphService(store).analyze_orphan_topics(min_degree=min_degree)


@pytest.mark.parametrize("limit", [-1, "many", True])
def test_analyze_orphan_topics_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        GraphService(store).analyze_orphan_topics(limit=limit)

