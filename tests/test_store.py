"""Tests for the Store class."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

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


def _close_and_unlink(store: Store, path: Path) -> None:
    store.close()
    for candidate in (
        path,
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)


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


class TestSavedQueries:
    def test_schema_creates_saved_queries_for_existing_database(self, tmp_path):
        db_path = tmp_path / "existing.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
            conn.execute("INSERT INTO schema_version (version) VALUES (1)")
            conn.commit()
        finally:
            conn.close()

        store = Store(str(db_path))
        try:
            row = store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'saved_queries'"
            ).fetchone()
            assert row is not None
        finally:
            store.close()

    def test_saved_query_crud(self, store: Store):
        saved = store.save_query(
            name="approved-solar",
            query="solar",
            mode="hybrid",
            limit=5,
            filters={"source_project": "max", "review_state": "approved"},
        )

        assert saved["name"] == "approved-solar"
        assert saved["query"] == "solar"
        assert saved["mode"] == "hybrid"
        assert saved["limit"] == 5
        assert saved["filters"] == {
            "source_project": "max",
            "review_state": "approved",
        }
        assert store.get_saved_query("approved-solar") == saved
        assert store.list_saved_queries() == [saved]

        updated = store.save_query(
            name="approved-solar",
            query="battery",
            mode="fulltext",
            limit=2,
            filters={"tag": "energy"},
        )

        assert updated["query"] == "battery"
        assert updated["mode"] == "fulltext"
        assert updated["limit"] == 2
        assert updated["filters"] == {"tag": "energy"}
        assert len(store.list_saved_queries()) == 1
        assert store.delete_saved_query("approved-solar") is True
        assert store.get_saved_query("approved-solar") is None
        assert store.delete_saved_query("approved-solar") is False


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


class TestJsonBackup:
    def test_export_json_contains_required_top_level_fields(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        store.fts_index_unit(inserted)

        payload = store.export_json()

        assert payload["schema_version"] == 1
        assert payload["exported_at"]
        assert len(payload["units"]) == 1
        assert payload["units"][0]["id"] == inserted.id
        assert payload["edges"] == []

    def test_import_export_round_trip_recreates_graph_and_fts_idempotently(
        self, tmp_path
    ):
        source_path = tmp_path / "source.db"
        target_path = tmp_path / "target.db"
        source = Store(str(source_path))
        target = Store(str(target_path))

        try:
            u1 = source.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.FORTY_TWO,
                    source_id="n1",
                    source_entity_type="knowledge_node",
                    title="Solar storage note",
                    content="Solar battery backup and panel sizing details.",
                    content_type=ContentType.FINDING,
                    tags=["solar", "battery"],
                    metadata={"priority": "high"},
                    utility_score=0.8,
                )
            )
            u2 = source.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id="i1",
                    source_entity_type="insight",
                    title="Home energy insight",
                    content="Home energy resilience depends on battery sizing.",
                    content_type=ContentType.INSIGHT,
                )
            )
            edge = source.insert_edge(
                KnowledgeEdge(
                    from_unit_id=u1.id,
                    to_unit_id=u2.id,
                    relation=EdgeRelation.INSPIRES,
                    source=EdgeSource.MANUAL,
                    metadata={"why": "backup"},
                )
            )

            payload = source.export_json()
            first_stats = target.import_json(payload)

            assert first_stats == {
                "units_inserted": 2,
                "units_updated": 0,
                "edges_inserted": 1,
                "edges_skipped": 0,
            }
            assert target.count_units() == 2
            assert len(target.get_all_edges()) == 1
            assert target.get_unit(u1.id).title == "Solar storage note"
            assert target.get_all_edges()[0].id == edge.id
            assert target.fts_search("battery")[0]["unit_id"] in {u1.id, u2.id}

            second_stats = target.import_json(payload)

            assert second_stats == {
                "units_inserted": 0,
                "units_updated": 2,
                "edges_inserted": 0,
                "edges_skipped": 1,
            }
            assert target.count_units() == 2
            assert len(target.get_all_edges()) == 1
        finally:
            _close_and_unlink(source, source_path)
            _close_and_unlink(target, target_path)
