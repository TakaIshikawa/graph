"""Tests for the Store class."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
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

    def test_update_unit_fields_merges_metadata_tags_and_reindexes_fts(
        self, store: Store, sample_unit: KnowledgeUnit
    ):
        sample_unit.metadata = {"review_state": "draft", "owner": "me"}
        sample_unit.tags = ["energy"]
        inserted = store.insert_unit(sample_unit)
        store.fts_index_unit(inserted)
        original = store.get_unit(inserted.id)

        updated = store.update_unit_fields(
            inserted.id,
            title="Battery sizing note",
            content="Battery storage sizing replaces old solar panel content.",
            content_type="idea",
            tags=["battery", "energy"],
            metadata={"review_state": "approved", "priority": "high"},
        )

        assert updated is not None
        assert updated.title == "Battery sizing note"
        assert updated.content_type == ContentType.IDEA
        assert updated.tags == ["energy", "battery"]
        assert updated.metadata == {
            "review_state": "approved",
            "owner": "me",
            "priority": "high",
        }
        assert str(updated.updated_at) != str(original.updated_at)
        assert store.fts_search("battery")[0]["unit_id"] == inserted.id
        assert store.fts_search("monocrystalline") == []

    def test_update_unit_fields_missing_unit(self, store: Store):
        assert store.update_unit_fields("missing", title="Nope") is None

    def test_pin_and_unpin_unit_preserves_existing_fields_and_metadata(
        self, store: Store, sample_unit: KnowledgeUnit
    ):
        sample_unit.metadata = {"review_state": "approved", "owner": "me"}
        sample_unit.tags = ["energy"]
        inserted = store.insert_unit(sample_unit)
        store.fts_index_unit(inserted)

        pinned = store.pin_unit(inserted.id, reason="evergreen")

        assert pinned is not None
        assert pinned.title == inserted.title
        assert pinned.content == inserted.content
        assert pinned.tags == ["energy"]
        assert pinned.metadata["review_state"] == "approved"
        assert pinned.metadata["owner"] == "me"
        assert pinned.metadata["pinned"] is True
        assert pinned.metadata["pin_reason"] == "evergreen"
        assert pinned.metadata["pinned_at"]
        assert store.fts_search(inserted.title)[0]["unit_id"] == inserted.id

        unpinned = store.unpin_unit(inserted.id)

        assert unpinned is not None
        assert unpinned.tags == ["energy"]
        assert unpinned.metadata == {"review_state": "approved", "owner": "me"}
        assert store.fts_search(inserted.title)[0]["unit_id"] == inserted.id

    def test_pin_and_unpin_unit_missing_unit(self, store: Store):
        assert store.pin_unit("missing", reason="nope") is None
        assert store.unpin_unit("missing") is None

    def test_merge_units_dry_run_reports_effects_without_writing(self, store: Store):
        target = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id="target",
                source_entity_type="manual",
                title="Target title",
                content="Target content",
                metadata={"owner": "target", "priority": "high"},
                tags=["solar", "shared"],
            )
        )
        source = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id="source",
                source_entity_type="manual",
                title="Source title",
                content="Source content",
                metadata={"owner": "source", "review_state": "approved"},
                tags=["shared", "battery"],
            )
        )
        neighbor = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="neighbor",
                source_entity_type="insight",
                title="Neighbor",
                content="Neighbor content",
            )
        )
        store.fts_index_unit(target)
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=source.id,
                to_unit_id=neighbor.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=target.id,
                to_unit_id=neighbor.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=neighbor.id,
                to_unit_id=source.id,
                relation=EdgeRelation.BUILDS_ON,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=source.id,
                to_unit_id=target.id,
                relation=EdgeRelation.INSPIRES,
            )
        )

        result = store.merge_units(source.id, target.id, dry_run=True)

        assert result["dry_run"] is True
        assert result["merged"] is False
        assert result["merged_tags"] == ["solar", "shared", "battery"]
        assert result["metadata_keys"] == ["owner", "review_state"]
        assert result["metadata_conflicts"] == ["owner"]
        assert result["rewired_edge_counts"] == {"incoming": 1, "outgoing": 0, "total": 1}
        assert len(result["skipped_duplicate_edges"]) == 1
        assert len(result["skipped_self_edges"]) == 1
        assert result["deleted_unit_id"] is None

        assert store.get_unit(source.id) is not None
        assert store.get_unit(target.id).metadata == {"owner": "target", "priority": "high"}
        assert len(store.get_all_edges()) == 4

    def test_merge_units_rewires_edges_merges_fields_and_deletes_source(self, store: Store):
        target = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id="target",
                source_entity_type="manual",
                title="Target title",
                content="Target searchable content",
                metadata={"owner": "target", "priority": "high"},
                tags=["solar", "shared"],
            )
        )
        source = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id="source",
                source_entity_type="manual",
                title="Source title",
                content="Source searchable content",
                metadata={"owner": "source", "review_state": "approved"},
                tags=["shared", "battery"],
            )
        )
        neighbor = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="neighbor",
                source_entity_type="insight",
                title="Neighbor",
                content="Neighbor content",
            )
        )
        store.fts_index_unit(target)
        store.fts_index_unit(source)
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=source.id,
                to_unit_id=neighbor.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=target.id,
                to_unit_id=neighbor.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=neighbor.id,
                to_unit_id=source.id,
                relation=EdgeRelation.BUILDS_ON,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=source.id,
                to_unit_id=target.id,
                relation=EdgeRelation.INSPIRES,
            )
        )

        result = store.merge_units(source.id, target.id)

        assert result["merged"] is True
        assert result["deleted_unit_id"] == source.id
        assert store.get_unit(source.id) is None
        merged = store.get_unit(target.id)
        assert merged.title == "Target title"
        assert merged.content == "Target searchable content"
        assert merged.tags == ["solar", "shared", "battery"]
        assert merged.metadata["owner"] == "target"
        assert merged.metadata["priority"] == "high"
        assert merged.metadata["review_state"] == "approved"
        assert merged.metadata["merged_from_units"][source.id]["metadata"] == {
            "owner": "source"
        }

        edges = store.get_all_edges()
        assert len(edges) == 2
        assert {(edge.from_unit_id, edge.to_unit_id, edge.relation) for edge in edges} == {
            (target.id, neighbor.id, EdgeRelation.RELATES_TO),
            (neighbor.id, target.id, EdgeRelation.BUILDS_ON),
        }
        assert store.fts_search("battery")[0]["unit_id"] == target.id
        assert store.fts_search("Source searchable") == []

    def test_merge_units_missing_and_same_ids(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)

        missing = store.merge_units("missing", inserted.id)

        assert missing["error"] == "unit_not_found"
        assert missing["missing_unit_ids"] == ["missing"]
        with pytest.raises(ValueError):
            store.merge_units(inserted.id, inserted.id)

    def test_get_pinned_units_filters_and_sorts_by_pinned_at(self, store: Store):
        older = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="older-pin",
                source_entity_type="insight",
                title="Older pinned",
                content="Older pinned content",
                content_type=ContentType.INSIGHT,
                tags=["solar", "workspace"],
                metadata={
                    "pinned": True,
                    "pinned_at": "2026-01-01T00:00:00+00:00",
                    "pin_reason": "older",
                },
            )
        )
        newer = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="newer-pin",
                source_entity_type="insight",
                title="Newer pinned",
                content="Newer pinned content",
                content_type=ContentType.INSIGHT,
                tags=["solar", "workspace"],
                metadata={
                    "pinned": True,
                    "pinned_at": "2026-01-02T00:00:00+00:00",
                    "pin_reason": "newer",
                },
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="other-project",
                source_entity_type="knowledge_node",
                title="Other project pinned",
                content="Other pinned content",
                content_type=ContentType.FINDING,
                tags=["solar", "workspace"],
                metadata={
                    "pinned": True,
                    "pinned_at": "2026-01-03T00:00:00+00:00",
                },
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="wrong-tag",
                source_entity_type="insight",
                title="Wrong tag pinned",
                content="Wrong tag pinned content",
                content_type=ContentType.INSIGHT,
                tags=["archive"],
                metadata={
                    "pinned": True,
                    "pinned_at": "2026-01-04T00:00:00+00:00",
                },
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="not-pinned",
                source_entity_type="insight",
                title="Not pinned",
                content="Not pinned content",
                content_type=ContentType.INSIGHT,
                tags=["solar", "workspace"],
            )
        )

        all_pinned = store.get_pinned_units(limit=2)
        assert [unit.title for unit in all_pinned] == [
            "Wrong tag pinned",
            "Other project pinned",
        ]

        filtered = store.get_pinned_units(
            source_project="max",
            content_type="insight",
            tag="workspace",
            limit=10,
        )

        assert [unit.id for unit in filtered] == [newer.id, older.id]
        assert filtered[0].metadata["pin_reason"] == "newer"

    def test_rename_tag_dry_run_and_execute_merges_reindexes_fts(self, store: Store):
        first = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="first",
                source_entity_type="insight",
                title="First tag note",
                content="Contains rename target",
                content_type=ContentType.INSIGHT,
                tags=["ai_agent", "workflow"],
            )
        )
        second = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="second",
                source_entity_type="insight",
                title="Second tag note",
                content="Contains merge target",
                content_type=ContentType.INSIGHT,
                tags=["ai_agent", "ai-agent", "review"],
            )
        )
        skipped = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="third",
                source_entity_type="knowledge_node",
                title="Skipped tag note",
                content="Different project",
                content_type=ContentType.FINDING,
                tags=["ai_agent"],
            )
        )
        for unit in [first, second, skipped]:
            store.fts_index_unit(unit)
        original_updated_at = str(store.get_unit(first.id).updated_at)  # type: ignore[union-attr]

        dry_run = store.rename_tag(
            "ai_agent",
            "ai-agent",
            dry_run=True,
            source_project="max",
            content_type="insight",
        )

        assert dry_run["dry_run"] is True
        assert dry_run["matched_count"] == 2
        assert dry_run["updated_count"] == 2
        assert dry_run["changed_count"] == 2
        assert set(dry_run["affected_unit_ids"]) == {first.id, second.id}
        assert {unit["id"] for unit in dry_run["changed_units"]} == {first.id, second.id}
        assert store.get_unit(first.id).tags == ["ai_agent", "workflow"]  # type: ignore[union-attr]
        assert str(store.get_unit(first.id).updated_at) == original_updated_at  # type: ignore[union-attr]

        result = store.rename_tag(
            "ai_agent",
            "ai-agent",
            source_project="max",
            content_type="insight",
        )

        assert result["dry_run"] is False
        assert result["matched_count"] == 2
        assert result["updated_count"] == 2
        assert result["changed_count"] == 2
        assert set(result["affected_unit_ids"]) == {first.id, second.id}
        assert store.get_unit(first.id).tags == ["ai-agent", "workflow"]  # type: ignore[union-attr]
        assert store.get_unit(second.id).tags == ["ai-agent", "review"]  # type: ignore[union-attr]
        assert store.get_unit(skipped.id).tags == ["ai_agent"]  # type: ignore[union-attr]
        assert str(store.get_unit(first.id).updated_at) != original_updated_at  # type: ignore[union-attr]
        assert {row["unit_id"] for row in store.fts_search("ai-agent")} >= {first.id, second.id}
        assert {row["unit_id"] for row in store.fts_search("ai_agent")} == {skipped.id}

    def test_apply_tags_to_units_dry_run_execute_dedupes_and_reindexes_fts(self, store: Store):
        first = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="first-bulk",
                source_entity_type="insight",
                title="Bulk first",
                content="Bulk searchable content",
                content_type=ContentType.INSIGHT,
                tags=["energy", "solar", "old"],
            )
        )
        second = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="second-bulk",
                source_entity_type="insight",
                title="Bulk second",
                content="Bulk searchable content",
                content_type=ContentType.INSIGHT,
                tags=["energy", "curated"],
            )
        )
        for unit in [first, second]:
            store.fts_index_unit(unit)
        original_updated_at = str(store.get_unit(first.id).updated_at)  # type: ignore[union-attr]

        dry_run = store.apply_tags_to_units(
            [first.id, first.id, second.id],
            add_tags=["curated", "reviewed", "reviewed"],
            remove_tags=["old"],
            dry_run=True,
        )

        assert dry_run["dry_run"] is True
        assert dry_run["matched_count"] == 2
        assert dry_run["changed_count"] == 2
        assert dry_run["changed_units"][0]["old_tags"] == ["energy", "solar", "old"]
        assert dry_run["changed_units"][0]["new_tags"] == [
            "energy",
            "solar",
            "curated",
            "reviewed",
        ]
        assert store.get_unit(first.id).tags == ["energy", "solar", "old"]  # type: ignore[union-attr]
        assert str(store.get_unit(first.id).updated_at) == original_updated_at  # type: ignore[union-attr]

        result = store.apply_tags_to_units(
            [first.id, second.id],
            add_tags=["curated", "reviewed"],
            remove_tags=["old"],
        )

        assert result["dry_run"] is False
        assert result["affected_count"] == 2
        assert store.get_unit(first.id).tags == [  # type: ignore[union-attr]
            "energy",
            "solar",
            "curated",
            "reviewed",
        ]
        assert store.get_unit(second.id).tags == ["energy", "curated", "reviewed"]  # type: ignore[union-attr]
        assert str(store.get_unit(first.id).updated_at) != original_updated_at  # type: ignore[union-attr]
        assert {row["unit_id"] for row in store.fts_search("reviewed")} == {
            first.id,
            second.id,
        }
        assert store.fts_search("old") == []


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

    def test_get_update_and_delete_edge_by_id(self, store: Store):
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
        edge = store.insert_edge(
            KnowledgeEdge(
                from_unit_id=u1.id,
                to_unit_id=u2.id,
                relation=EdgeRelation.BUILDS_ON,
                source=EdgeSource.INFERRED,
                metadata={"old": "value"},
            )
        )

        fetched = store.get_edge(edge.id)
        assert fetched is not None
        assert fetched.id == edge.id

        updated = store.update_edge_fields(
            edge.id,
            relation="inspires",
            weight=0.4,
            source="manual",
            metadata={"new": "value"},
        )

        assert updated is not None
        assert updated.relation == EdgeRelation.INSPIRES
        assert updated.weight == 0.4
        assert updated.source == EdgeSource.MANUAL
        assert updated.metadata == {"old": "value", "new": "value"}
        backlink = store.get_backlinks(u2.id)["links"][0]
        assert backlink["edge"].id == edge.id
        assert backlink["relation"] == "inspires"
        assert backlink["edge"].weight == 0.4
        assert store.update_edge_fields("missing", relation="relates_to") is None

        assert store.delete_edge(edge.id) == {"edge_id": edge.id, "deleted": True}
        assert store.get_unit(u1.id) is not None
        assert store.get_unit(u2.id) is not None
        assert store.get_all_edges() == []
        assert store.delete_edge(edge.id) == {"edge_id": edge.id, "deleted": False}

    def test_import_edges_csv_validates_dry_runs_and_skips_duplicates(
        self, store: Store, tmp_path
    ):
        u1 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id="manual-1",
                source_entity_type="manual",
                title="Unit 1",
                content="Content 1",
            )
        )
        u2 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id="manual-2",
                source_entity_type="manual",
                title="Unit 2",
                content="Content 2",
            )
        )
        u3 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id="manual-3",
                source_entity_type="manual",
                title="Unit 3",
                content="Content 3",
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=u1.id,
                to_unit_id=u2.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
        csv_path = tmp_path / "edges.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "from_unit_id,to_unit_id,relation,weight,source,metadata_json",
                    f'{u1.id},{u2.id},relates_to,0.8,manual,{{"note":"duplicate"}}',
                    f'{u2.id},{u3.id},inspires,0.5,source,{{"note":"valid"}}',
                    f'{u2.id},{u3.id},inspires,0.5,source,{{"note":"in-file duplicate"}}',
                    f"missing,{u3.id},unknown,nope,bad,[]",
                ]
            )
            + "\n"
        )

        dry_run = store.import_edges_csv(csv_path, dry_run=True)

        assert dry_run["dry_run"] is True
        assert dry_run["inserted"] == 1
        assert dry_run["skipped_existing"] == 2
        assert dry_run["inserted_rows"] == [3]
        assert dry_run["skipped_existing_rows"] == [2, 4]
        assert dry_run["invalid"][0]["row_number"] == 5
        assert "from_unit_id not found: missing" in dry_run["invalid"][0]["reasons"]
        assert len(store.get_all_edges()) == 1

        applied = store.import_edges_csv(csv_path)

        assert applied["dry_run"] is False
        assert applied["inserted"] == 1
        assert applied["skipped_existing"] == 2
        edges = store.get_all_edges()
        assert len(edges) == 2
        imported = next(edge for edge in edges if edge.to_unit_id == u3.id)
        assert imported.relation == EdgeRelation.INSPIRES
        assert imported.weight == 0.5
        assert imported.source == EdgeSource.SOURCE
        assert imported.metadata == {"note": "valid"}

    def test_delete_unit_removes_fts_row_and_related_edges(self, store: Store):
        u1 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id="manual-1",
                source_entity_type="manual",
                title="Manual solar note",
                content="Solar content to delete",
            )
        )
        u2 = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id="manual-2",
                source_entity_type="manual",
                title="Manual battery note",
                content="Battery content",
            )
        )
        store.fts_index_unit(u1)
        store.fts_index_unit(u2)
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=u1.id,
                to_unit_id=u2.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )

        stats = store.delete_unit(u1.id)

        assert stats == {"unit_id": u1.id, "deleted": True, "edges_deleted": 1}
        assert store.get_unit(u1.id) is None
        assert store.get_all_edges() == []
        assert store.fts_search("solar") == []
        assert store.fts_search("battery")[0]["unit_id"] == u2.id
        assert store.delete_unit(u1.id) == {
            "unit_id": u1.id,
            "deleted": False,
            "edges_deleted": 0,
        }


class TestBacklinks:
    def test_get_backlinks_expands_incoming_and_outgoing_units(self, store: Store):
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
                source=EdgeSource.MANUAL,
                metadata={"why": "supporting evidence"},
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=u2.id,
                to_unit_id=u3.id,
                relation=EdgeRelation.INSPIRES,
            )
        )

        result = store.get_backlinks(u2.id, direction="both")

        assert result["center"].id == u2.id
        assert {(link["direction"], link["unit"].title) for link in result["links"]} == {
            ("incoming", "Unit 1"),
            ("outgoing", "Unit 3"),
        }
        incoming = next(link for link in result["links"] if link["direction"] == "incoming")
        assert incoming["relation"] == "builds_on"
        assert incoming["edge"].metadata == {"why": "supporting evidence"}

    def test_get_backlinks_filters_direction_relation_and_limit(self, store: Store):
        center = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="center",
                source_entity_type="knowledge_node",
                title="Center",
                content="Center content",
            )
        )
        incoming = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="incoming",
                source_entity_type="insight",
                title="Incoming",
                content="Incoming content",
            )
        )
        outgoing = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="outgoing",
                source_entity_type="insight",
                title="Outgoing",
                content="Outgoing content",
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=incoming.id,
                to_unit_id=center.id,
                relation=EdgeRelation.BUILDS_ON,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=center.id,
                to_unit_id=outgoing.id,
                relation=EdgeRelation.INSPIRES,
            )
        )

        incoming_only = store.get_backlinks(center.id, direction="incoming")
        assert [link["unit"].title for link in incoming_only["links"]] == ["Incoming"]

        inspires_only = store.get_backlinks(center.id, direction="both", relation="inspires")
        assert [link["unit"].title for link in inspires_only["links"]] == ["Outgoing"]

        limited = store.get_backlinks(center.id, limit=1)
        assert len(limited["links"]) == 1

    def test_get_backlinks_missing_unit(self, store: Store):
        assert store.get_backlinks("missing") == {"center": None, "links": []}

    def test_get_backlinks_filters_source_unit_and_orders_by_weight_then_updated_at(
        self, store: Store
    ):
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        center = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="center-filtered",
                source_entity_type="knowledge_node",
                title="Center",
                content="Center content",
            )
        )
        older = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="older",
                source_entity_type="insight",
                title="Older matching source",
                content="Older source content",
                content_type=ContentType.INSIGHT,
                tags=["energy"],
                updated_at=base_time,
            )
        )
        newer = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="newer",
                source_entity_type="insight",
                title="Newer matching source",
                content="Newer source content",
                content_type=ContentType.INSIGHT,
                tags=["energy"],
                updated_at=base_time + timedelta(days=1),
            )
        )
        heavier = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="heavier",
                source_entity_type="insight",
                title="Heavier matching source",
                content="Heavier source content",
                content_type=ContentType.INSIGHT,
                tags=["energy"],
                updated_at=base_time - timedelta(days=1),
            )
        )
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.PRESENCE,
                source_id="wrong-project",
                source_entity_type="knowledge_item",
                title="Wrong project source",
                content="Wrong source content",
                content_type=ContentType.ARTIFACT,
                tags=["energy"],
            )
        )

        for unit, weight in ((older, 0.6), (newer, 0.6), (heavier, 0.9)):
            store.insert_edge(
                KnowledgeEdge(
                    from_unit_id=unit.id,
                    to_unit_id=center.id,
                    relation=EdgeRelation.REFERENCES,
                    weight=weight,
                )
            )

        result = store.get_backlinks(
            center.id,
            direction="incoming",
            relation="references",
            source_project="max",
            content_type="insight",
            tag="energy",
            limit=3,
        )

        assert [link["unit"].title for link in result["links"]] == [
            "Heavier matching source",
            "Newer matching source",
            "Older matching source",
        ]


class TestFTS:
    def test_fts_search(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        store.fts_index_unit(inserted)

        results = store.fts_search("solar")
        assert len(results) == 1
        assert results[0]["unit_id"] == inserted.id
        assert results[0]["snippet"]
        assert "[Solar]" in results[0]["snippet"]

    def test_fts_no_results(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        store.fts_index_unit(inserted)

        results = store.fts_search("quantum")
        assert len(results) == 0

    def test_fts_fallback_on_invalid_syntax(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        store.fts_index_unit(inserted)

        # Invalid FTS5 syntax should fallback to LIKE
        results = store.fts_search('"solar')
        assert len(results) == 1
        assert results[0]["unit_id"] == inserted.id
        assert results[0]["snippet"]

    def test_integrity_helpers_find_and_repair_fts_drift(
        self, store: Store, sample_unit: KnowledgeUnit
    ):
        missing = store.insert_unit(sample_unit)
        store.conn.execute(
            "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
            ("deleted-unit", "Deleted", "stale content", "stale"),
        )
        store.conn.commit()

        assert store.find_units_missing_fts_rows()["count"] == 1
        assert store.find_stale_fts_rows()["count"] == 1

        repair = store.repair_fts_index_integrity()

        assert repair == {"fts_rows_inserted": 1, "fts_rows_deleted": 1}
        assert store.find_units_missing_fts_rows()["count"] == 0
        assert store.find_stale_fts_rows()["count"] == 0
        assert store.fts_search("monocrystalline")[0]["unit_id"] == missing.id

    def test_integrity_helpers_find_structural_issues(
        self, store: Store, sample_unit: KnowledgeUnit
    ):
        unit = store.insert_unit(sample_unit)
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=unit.id,
                to_unit_id=unit.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
        store.conn.execute("PRAGMA foreign_keys=OFF")
        store.conn.execute(
            """INSERT INTO edges
               (id, from_unit_id, to_unit_id, relation, weight, source, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "dangling-edge",
                unit.id,
                "missing-unit",
                EdgeRelation.RELATES_TO.value,
                1.0,
                EdgeSource.MANUAL.value,
                "{}",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        store.conn.execute(
            "UPDATE knowledge_units SET metadata = ?, title = ? WHERE id = ?",
            ("{bad json", "   ", unit.id),
        )
        store.conn.commit()
        store.conn.execute("PRAGMA foreign_keys=ON")

        assert store.find_dangling_edges()["examples"][0]["edge_id"] == "dangling-edge"
        assert store.find_self_loop_edges()["count"] == 1
        assert store.find_invalid_json_rows()["count"] == 1
        blank = store.find_blank_units()["examples"][0]
        assert blank["unit_id"] == unit.id
        assert blank["blank_title"] is True


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

    def test_freshness_report_counts_recent_units_and_marks_stale(self, store: Store):
        now = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)
        recent = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="recent",
                source_entity_type="insight",
                title="Recent",
                content="Recently ingested unit",
            )
        )
        old = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="old",
                source_entity_type="insight",
                title="Old",
                content="Older ingested unit",
            )
        )
        store.conn.execute(
            "UPDATE knowledge_units SET ingested_at = ? WHERE id = ?",
            ((now - timedelta(days=3)).isoformat(), recent.id),
        )
        store.conn.execute(
            "UPDATE knowledge_units SET ingested_at = ? WHERE id = ?",
            ((now - timedelta(days=8)).isoformat(), old.id),
        )
        store.conn.commit()
        store.upsert_sync_state(
            SyncState(
                source_project="max",
                source_entity_type="insight",
                last_sync_at=now - timedelta(days=8),
                items_synced=2,
            )
        )

        report = store.freshness_report(
            [("max", "insight"), ("presence", "knowledge_item")],
            days=7,
            now=now,
        )

        assert report[0] == {
            "source_project": "max",
            "source_entity_type": "insight",
            "last_sync_at": "2026-04-16T12:00:00+00:00",
            "age_days": 8.0,
            "recent_unit_count": 1,
            "total_unit_count": 2,
            "stale": True,
        }
        assert report[1] == {
            "source_project": "presence",
            "source_entity_type": "knowledge_item",
            "last_sync_at": None,
            "age_days": None,
            "recent_unit_count": 0,
            "total_unit_count": 0,
            "stale": True,
        }

        wider_report = store.freshness_report([("max", "insight")], days=10, now=now)
        assert wider_report[0]["recent_unit_count"] == 2
        assert wider_report[0]["stale"] is False


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

    def test_schema_adds_embedding_timestamp_for_existing_database(self, tmp_path):
        db_path = tmp_path / "existing-embeddings.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE schema_version (version INTEGER NOT NULL);
                INSERT INTO schema_version (version) VALUES (1);
                CREATE TABLE knowledge_units (
                    id TEXT PRIMARY KEY,
                    source_project TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_entity_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_type TEXT NOT NULL DEFAULT 'insight',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    tags TEXT NOT NULL DEFAULT '[]',
                    confidence REAL,
                    utility_score REAL,
                    embedding BLOB,
                    created_at TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(source_project, source_id, source_entity_type)
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

        store = Store(str(db_path))
        try:
            columns = {
                row["name"] for row in store.conn.execute("PRAGMA table_info(knowledge_units)")
            }
            assert "embedding_updated_at" in columns
            version = store.conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version == 2
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

    def test_saved_query_export_import_round_trip_and_updates_by_name(self, tmp_path):
        source = Store(str(tmp_path / "source.db"))
        target = Store(str(tmp_path / "target.db"))
        try:
            source.save_query(
                name="approved-solar",
                query="solar",
                mode="hybrid",
                limit=5,
                filters={"source_project": "max", "review_state": "approved"},
            )
            source.save_query(
                name="battery",
                query="battery storage",
                mode="fulltext",
                limit=3,
                filters={"tag": "energy"},
            )

            payload = source.export_saved_queries()

            assert payload["schema_version"] == 1
            assert [query["name"] for query in payload["queries"]] == [
                "approved-solar",
                "battery",
            ]

            target.save_query(
                name="approved-solar",
                query="outdated",
                mode="semantic",
                limit=1,
                filters={"tag": "old"},
            )

            stats = target.import_saved_queries(payload)

            assert stats == {"inserted": 1, "updated": 1, "skipped": 0}
            assert target.list_saved_queries() == payload["queries"]
            assert target.import_saved_queries(payload) == {
                "inserted": 0,
                "updated": 0,
                "skipped": 2,
            }
        finally:
            source.close()
            target.close()

    def test_saved_query_import_rejects_invalid_schema_version(self, store: Store):
        with pytest.raises(ValueError, match="Unsupported saved queries schema_version 999"):
            store.import_saved_queries({"schema_version": 999, "queries": []})


class TestEmbedding:
    def test_update_and_get_embeddings(self, store: Store, sample_unit: KnowledgeUnit):
        inserted = store.insert_unit(sample_unit)
        original = store.get_unit(inserted.id)
        import struct

        embedding = [0.1, 0.2, 0.3]
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        store.update_embedding(inserted.id, blob)

        row = store.conn.execute(
            "SELECT embedding_updated_at, updated_at FROM knowledge_units WHERE id = ?",
            (inserted.id,),
        ).fetchone()
        assert row["embedding_updated_at"] is not None
        assert row["updated_at"] == original.updated_at.isoformat()

        results = store.get_units_with_embeddings()
        assert len(results) == 1
        unit, emb_bytes = results[0]
        assert unit.id == inserted.id
        restored = list(struct.unpack(f"{len(emb_bytes) // 4}f", emb_bytes))
        assert len(restored) == 3
        assert abs(restored[0] - 0.1) < 1e-6

    def test_embedding_status_counts_missing_fresh_and_stale(
        self, store: Store, sample_unit: KnowledgeUnit
    ):
        fresh = store.insert_unit(sample_unit)
        stale = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="stale",
                source_entity_type="insight",
                title="Stale",
                content="Old content",
            )
        )
        missing = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="missing",
                source_entity_type="insight",
                title="Missing",
                content="No embedding",
            )
        )
        store.update_embedding(fresh.id, b"fresh")
        store.update_embedding(stale.id, b"stale")
        store.update_unit_fields(stale.id, content="New content")

        assert store.get_embedding_status() == {
            "total": 3,
            "missing": 1,
            "fresh": 1,
            "stale": 1,
            "percent_fresh": 33.33,
        }
        assert store.get_embedding_status(source_project="max") == {
            "total": 2,
            "missing": 1,
            "fresh": 0,
            "stale": 1,
            "percent_fresh": 0.0,
        }
        assert store.get_embedding_status_groups("source_project") == [
            {
                "source_project": "forty_two",
                "total": 1,
                "missing": 0,
                "fresh": 1,
                "stale": 0,
                "percent_fresh": 100.0,
            },
            {
                "source_project": "max",
                "total": 2,
                "missing": 1,
                "fresh": 0,
                "stale": 1,
                "percent_fresh": 0.0,
            },
        ]
        assert store.get_embedding_status_matrix(source_project="max") == [
            {
                "source_project": "max",
                "content_type": "insight",
                "total": 2,
                "missing": 1,
                "fresh": 0,
                "stale": 1,
                "percent_fresh": 0.0,
            }
        ]
        refresh = store.get_embedding_refresh_status(source_project="max", limit=5)
        assert {item["id"]: item["reason"] for item in refresh} == {
            stale.id: "stale_embedding",
            missing.id: "missing_embedding",
        }
        assert [u.id for u in store.get_units_for_embedding_refresh()] == [missing.id]
        assert {u.id for u in store.get_units_for_embedding_refresh(stale_only=True)} == {
            stale.id,
            missing.id,
        }


class TestJsonBackup:
    def test_export_json_contains_required_top_level_fields(
        self, store: Store, sample_unit: KnowledgeUnit
    ):
        inserted = store.insert_unit(sample_unit)
        store.fts_index_unit(inserted)

        payload = store.export_json()

        assert payload["schema_version"] == 2
        assert payload["exported_at"]
        assert len(payload["units"]) == 1
        assert payload["units"][0]["id"] == inserted.id
        assert payload["edges"] == []

    def test_import_export_round_trip_recreates_graph_and_fts_idempotently(self, tmp_path):
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
