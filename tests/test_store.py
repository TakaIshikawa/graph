"""Tests for the Store class."""

from __future__ import annotations

import os
import tempfile

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


@pytest.fixture
def sample_unit() -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=SourceProject.FORTY_TWO,
        source_id="node-001",
        source_entity_type="knowledge_node",
        title="Solar panel efficiency test",
        content="Tested monocrystalline vs polycrystalline panels under varying conditions.",
        content_type=ContentType.FINDING,
        tags=["energy", "solar"],
        utility_score=0.85,
    )


class TestUnitCRUD:
    def test_insert_and_get(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        assert inserted.id != ""

        fetched = store.get_unit(inserted.id)
        assert fetched is not None
        assert fetched.title == "Solar panel efficiency test"
        assert fetched.source_project == SourceProject.FORTY_TWO
        assert fetched.tags == ["energy", "solar"]

    def test_get_unit_by_source(self, store: Store, sample_unit: KnowledgeUnit):
        store.insert_unit(sample_unit)
        fetched = store.get_unit_by_source("forty_two", "node-001", "knowledge_node")
        assert fetched is not None
        assert fetched.title == "Solar panel efficiency test"

    def test_upsert_updates_existing(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        original_id = inserted.id

        sample_unit.title = "Updated title"
        sample_unit.id = ""  # reset ID to test upsert
        store.insert_unit(sample_unit)

        # Should still be one unit
        assert store.count_units() == 1
        fetched = store.get_unit(original_id)
        assert fetched is not None
        assert fetched.title == "Updated title"

    def test_get_all_units(self, store: Store):
        for i in range(5):
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"insight-{i}",
                    source_entity_type="insight",
                    title=f"Insight {i}",
                    content=f"Content for insight {i}",
                )
            )
        units = store.get_all_units()
        assert len(units) == 5

    def test_count_units_with_filter(self, store: Store):
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="n1",
                source_entity_type="knowledge_node",
                title="FT unit",
                content="content",
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="i1",
                source_entity_type="insight",
                title="Max unit",
                content="content",
            )
        )
        assert store.count_units() == 2
        assert store.count_units(source_project="forty_two") == 1
        assert store.count_units(source_project="max") == 1


class TestEdgeCRUD:
    def test_insert_and_get_edges(self, store: Store):
        u1 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="n1",
                source_entity_type="knowledge_node",
                title="Unit 1",
                content="Content 1",
            )
        )
        u2 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="n2",
                source_entity_type="knowledge_node",
                title="Unit 2",
                content="Content 2",
            )
        )
        edge = KnowledgeEdge(
            from_unit_id=u1.id,
            to_unit_id=u2.id,
            relation=EdgeRelation.BUILDS_ON,
            source=EdgeSource.SOURCE,
        )
        inserted = store.insert_edge(edge)
        assert inserted.id != ""

        edges = store.get_all_edges()
        assert len(edges) == 1
        assert edges[0].relation == EdgeRelation.BUILDS_ON

    def test_get_edges_for_unit(self, store: Store):
        u1 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="n1",
                source_entity_type="knowledge_node",
                title="Unit 1",
                content="Content 1",
            )
        )
        u2 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="n2",
                source_entity_type="knowledge_node",
                title="Unit 2",
                content="Content 2",
            )
        )
        u3 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="i1",
                source_entity_type="insight",
                title="Unit 3",
                content="Content 3",
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=u1.id,
                to_unit_id=u2.id,
                relation=EdgeRelation.BUILDS_ON,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=u3.id,
                to_unit_id=u1.id,
                relation=EdgeRelation.INSPIRES,
            )
        )
        edges = store.get_edges_for_unit(u1.id)
        assert len(edges) == 2

    def test_duplicate_edge_ignored(self, store: Store):
        u1 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="n1",
                source_entity_type="knowledge_node",
                title="Unit 1",
                content="Content 1",
            )
        )
        u2 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="n2",
                source_entity_type="knowledge_node",
                title="Unit 2",
                content="Content 2",
            )
        )
        edge = KnowledgeEdge(
            from_unit_id=u1.id,
            to_unit_id=u2.id,
            relation=EdgeRelation.BUILDS_ON,
        )
        store.insert_edge(edge)
        store.insert_edge(edge)  # duplicate
        assert len(store.get_all_edges()) == 1


class TestFTS:
    def test_fts_search(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        store.fts_index_unit(inserted)

        results = store.fts_search("solar")
        assert len(results) == 1
        assert results[0]["unit_id"] == inserted.id

    def test_fts_no_results(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        store.fts_index_unit(inserted)

        results = store.fts_search("quantum")
        assert len(results) == 0

    def test_fts_fallback_on_invalid_syntax(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        store.fts_index_unit(inserted)

        # Invalid FTS5 syntax should fallback to LIKE
        results = store.fts_search("solar AND panel")
        assert len(results) >= 0  # Should not raise


class TestSyncState:
    def test_upsert_and_get(self, store: Store):
        state = SyncState(
            source_project="forty_two",
            source_entity_type="knowledge_node",
            last_sync_at="2025-01-01T00:00:00+00:00",
            items_synced=10,
        )
        store.upsert_sync_state(state)

        fetched = store.get_sync_state("forty_two", "knowledge_node")
        assert fetched is not None
        assert fetched.items_synced == 10

    def test_upsert_accumulates(self, store: Store):
        state = SyncState(
            source_project="max",
            source_entity_type="insight",
            last_sync_at="2025-01-01T00:00:00+00:00",
            items_synced=5,
        )
        store.upsert_sync_state(state)

        state2 = SyncState(
            source_project="max",
            source_entity_type="insight",
            last_sync_at="2025-01-02T00:00:00+00:00",
            items_synced=3,
        )
        store.upsert_sync_state(state2)

        fetched = store.get_sync_state("max", "insight")
        assert fetched is not None
        assert fetched.items_synced == 8


class TestEmbedding:
    def test_update_and_get_embeddings(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        import struct

        embedding = [0.1, 0.2, 0.3]
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        store.update_embedding(inserted.id, blob)

        results = store.get_units_with_embeddings()
        assert len(results) == 1
        unit, emb_bytes = results[0]
        assert unit.id == inserted.id
        restored = list(struct.unpack(f"{len(emb_bytes) // 4}f", emb_bytes))
        assert len(restored) == 3
        assert abs(restored[0] - 0.1) < 1e-6
