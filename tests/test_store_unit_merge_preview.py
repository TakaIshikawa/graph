from __future__ import annotations

import os
import tempfile

import pytest

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


def unit(unit_id: str, *, tags: list[str] | None = None, metadata: dict | None = None):
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        tags=tags or [],
        metadata=metadata or {},
    )


def test_preview_unit_merge_reports_metadata_conflicts_and_tag_union(store: Store):
    primary = store.insert_unit(unit("primary", tags=["solar"], metadata={"owner": "a"}))
    duplicate = store.insert_unit(
        unit("duplicate", tags=["solar", "storage"], metadata={"owner": "b", "status": "new"})
    )

    preview = store.preview_unit_merge(primary.id, [duplicate.id])

    assert preview["mergeable"] is True
    assert preview["merged_title"] == "Title primary"
    assert preview["merged_content"] == "Content primary"
    assert preview["combined_tags"] == ["solar", "storage"]
    assert preview["metadata_conflicts"] == [
        {
            "key": "owner",
            "primary_value": "a",
            "duplicate_unit_id": duplicate.id,
            "duplicate_value": "b",
        }
    ]
    assert preview["units_removed"] == [duplicate.id]


def test_preview_unit_merge_reports_rewired_and_collapsed_edges(store: Store):
    primary = store.insert_unit(unit("primary"))
    duplicate = store.insert_unit(unit("duplicate"))
    neighbor = store.insert_unit(unit("neighbor"))
    other = store.insert_unit(unit("other"))
    store.insert_edge(
        KnowledgeEdge(
            id="duplicate-edge",
            from_unit_id=duplicate.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            id="existing-edge",
            from_unit_id=primary.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            id="rewire-edge",
            from_unit_id=other.id,
            to_unit_id=duplicate.id,
            relation=EdgeRelation.BUILDS_ON,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            id="self-edge",
            from_unit_id=duplicate.id,
            to_unit_id=primary.id,
            relation=EdgeRelation.INSPIRES,
        )
    )

    preview = store.preview_unit_merge(primary.id, [duplicate.id])

    assert [edge["edge_id"] for edge in preview["duplicate_edges_collapsed"]] == [
        "duplicate-edge"
    ]
    assert preview["edges_rewired"] == [
        {
            "edge_id": "rewire-edge",
            "from_unit_id": other.id,
            "to_unit_id": duplicate.id,
            "relation": EdgeRelation.BUILDS_ON,
            "new_from_unit_id": other.id,
            "new_to_unit_id": primary.id,
        }
    ]
    assert [edge["edge_id"] for edge in preview["self_edges_skipped"]] == ["self-edge"]


def test_preview_unit_merge_reports_missing_ids(store: Store):
    primary = store.insert_unit(unit("primary"))

    preview = store.preview_unit_merge(primary.id, ["missing"])

    assert preview["mergeable"] is False
    assert preview["error"] == "unit_not_found"
    assert preview["missing_unit_ids"] == ["missing"]


def test_preview_unit_merge_does_not_mutate_units_edges_or_fts(store: Store):
    primary = store.insert_unit(unit("primary", tags=["a"], metadata={"owner": "primary"}))
    duplicate = store.insert_unit(unit("duplicate", tags=["b"], metadata={"owner": "duplicate"}))
    store.fts_index_unit(primary)
    store.fts_index_unit(duplicate)
    store.insert_edge(
        KnowledgeEdge(
            id="edge",
            from_unit_id=duplicate.id,
            to_unit_id=primary.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    before_units = {item.id: (item.tags, item.metadata) for item in store.get_all_units()}
    before_edges = [(edge.id, edge.from_unit_id, edge.to_unit_id) for edge in store.get_all_edges()]
    before_fts = store.fts_search("Content duplicate")

    store.preview_unit_merge(primary.id, [duplicate.id])

    after_units = {item.id: (item.tags, item.metadata) for item in store.get_all_units()}
    after_edges = [(edge.id, edge.from_unit_id, edge.to_unit_id) for edge in store.get_all_edges()]
    assert after_units == before_units
    assert after_edges == before_edges
    assert store.fts_search("Content duplicate") == before_fts
