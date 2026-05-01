from __future__ import annotations

import math
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


def _insert_unit(
    store: Store,
    unit_id: str,
    *,
    source_project: SourceProject,
    content_type: ContentType,
    title: str,
    tags: list[str] | None = None,
):
    return store.insert_unit(
        KnowledgeUnit(
            id=unit_id,
            source_project=source_project,
            source_id=unit_id,
            source_entity_type="note",
            title=title,
            content=f"{title} content",
            content_type=content_type,
            tags=tags or [],
        )
    )


def _insert_edge(
    store: Store,
    from_unit_id: str,
    to_unit_id: str,
    *,
    weight: float = 1.0,
):
    return store.insert_edge(
        KnowledgeEdge(
            from_unit_id=from_unit_id,
            to_unit_id=to_unit_id,
            relation=EdgeRelation.RELATES_TO,
            weight=weight,
            source=EdgeSource.MANUAL,
        )
    )


def test_assortativity_empty_graph_is_stable(store: Store):
    result = GraphService(store).analyze_assortativity()

    assert result == {
        "node_count": 0,
        "edge_count": 0,
        "source_project_assortativity": 0.0,
        "content_type_assortativity": 0.0,
        "tag_similarity": 0.0,
        "baseline_tag_similarity": 0.0,
        "top_cross_source_edges": [],
    }


def test_assortativity_homogeneous_graph_handles_degenerate_categories(store: Store):
    _insert_unit(
        store,
        "a",
        source_project=SourceProject.MAX,
        content_type=ContentType.INSIGHT,
        title="Alpha",
        tags=["energy"],
    )
    _insert_unit(
        store,
        "b",
        source_project=SourceProject.MAX,
        content_type=ContentType.INSIGHT,
        title="Beta",
        tags=["energy"],
    )
    _insert_unit(
        store,
        "c",
        source_project=SourceProject.MAX,
        content_type=ContentType.INSIGHT,
        title="Gamma",
        tags=["energy"],
    )
    _insert_edge(store, "a", "b")
    _insert_edge(store, "b", "c")

    result = GraphService(store).analyze_assortativity()

    assert result["node_count"] == 3
    assert result["edge_count"] == 2
    assert result["source_project_assortativity"] == 1.0
    assert result["content_type_assortativity"] == 1.0
    assert result["tag_similarity"] == 1.0
    assert result["baseline_tag_similarity"] == 1.0
    assert result["top_cross_source_edges"] == []


def test_assortativity_mixed_graph_reports_categories_and_cross_source_edges(
    store: Store,
):
    _insert_unit(
        store,
        "a",
        source_project=SourceProject.MAX,
        content_type=ContentType.INSIGHT,
        title="Alpha",
        tags=["energy", "solar"],
    )
    _insert_unit(
        store,
        "b",
        source_project=SourceProject.MAX,
        content_type=ContentType.INSIGHT,
        title="Beta",
        tags=["energy", "solar", "storage"],
    )
    _insert_unit(
        store,
        "c",
        source_project=SourceProject.FORTY_TWO,
        content_type=ContentType.FINDING,
        title="Gamma",
        tags=["energy", "grid"],
    )
    _insert_unit(
        store,
        "d",
        source_project=SourceProject.PRESENCE,
        content_type=ContentType.ARTIFACT,
        title="Delta",
        tags=["writing"],
    )
    _insert_edge(store, "a", "b", weight=0.8)
    _insert_edge(store, "b", "c", weight=0.6)
    _insert_edge(store, "a", "d", weight=0.9)

    result = GraphService(store).analyze_assortativity()

    assert result["node_count"] == 4
    assert result["edge_count"] == 3
    assert not math.isnan(result["source_project_assortativity"])
    assert not math.isnan(result["content_type_assortativity"])
    assert result["tag_similarity"] == pytest.approx((2 / 3 + 1 / 4 + 0) / 3)
    assert result["baseline_tag_similarity"] == pytest.approx(1 / 9)
    assert result["top_cross_source_edges"] == [
        {
            "from_unit_id": "a",
            "from_title": "Alpha",
            "from_source_project": "max",
            "to_unit_id": "d",
            "to_title": "Delta",
            "to_source_project": "presence",
            "relation": "relates_to",
            "weight": 0.9,
        },
        {
            "from_unit_id": "b",
            "from_title": "Beta",
            "from_source_project": "max",
            "to_unit_id": "c",
            "to_title": "Gamma",
            "to_source_project": "forty_two",
            "relation": "relates_to",
            "weight": 0.6,
        },
    ]


def test_assortativity_cross_source_edges_have_deterministic_order(store: Store):
    _insert_unit(
        store,
        "a",
        source_project=SourceProject.MAX,
        content_type=ContentType.INSIGHT,
        title="Alpha",
    )
    _insert_unit(
        store,
        "b",
        source_project=SourceProject.FORTY_TWO,
        content_type=ContentType.FINDING,
        title="Beta",
    )
    _insert_unit(
        store,
        "c",
        source_project=SourceProject.PRESENCE,
        content_type=ContentType.ARTIFACT,
        title="Gamma",
    )
    _insert_edge(store, "c", "a", weight=0.5)
    _insert_edge(store, "a", "b", weight=0.5)

    result = GraphService(store).analyze_assortativity()

    assert [
        (edge["from_unit_id"], edge["to_unit_id"])
        for edge in result["top_cross_source_edges"]
    ] == [("a", "b"), ("c", "a")]
