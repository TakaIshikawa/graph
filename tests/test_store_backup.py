from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

import pytest

from graph.store.backup import export_store_backup
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    yield store
    store.close()


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _unit(unit_id: str, source_id: str, created_at: datetime) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=source_id,
        source_entity_type="insight",
        title=f"Unit {unit_id}",
        content=f"Content for {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata={"rank": source_id, "nested": {"flag": True}},
        tags=["backup", source_id],
        confidence=0.7,
        utility_score=0.9,
        created_at=created_at,
        updated_at=created_at,
    )


def test_export_store_backup_exports_units_edges_sync_state_and_metadata(store: Store):
    later = store.insert_unit(_unit("unit-b", "b", _dt("2026-01-02T00:00:00+00:00")))
    earlier = store.insert_unit(_unit("unit-a", "a", _dt("2026-01-01T00:00:00+00:00")))
    store.insert_edge(
        KnowledgeEdge(
            id="edge-b",
            from_unit_id=later.id,
            to_unit_id=earlier.id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.SOURCE,
            weight=0.25,
            metadata={"why": "source order"},
            created_at=_dt("2026-01-04T00:00:00+00:00"),
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            id="edge-a",
            from_unit_id=earlier.id,
            to_unit_id=later.id,
            relation=EdgeRelation.BUILDS_ON,
            source=EdgeSource.MANUAL,
            weight=0.5,
            metadata={"why": "backup"},
            created_at=_dt("2026-01-03T00:00:00+00:00"),
        )
    )
    store.upsert_sync_state(
        SyncState(
            source_project="max",
            source_entity_type="insight",
            last_sync_at=_dt("2026-01-05T00:00:00+00:00"),
            last_source_id="b",
            items_synced=2,
        )
    )
    store.upsert_sync_state(
        SyncState(
            source_project="forty_two",
            source_entity_type="knowledge_node",
            last_sync_at=_dt("2026-01-06T00:00:00+00:00"),
            last_source_id="node-1",
            items_synced=1,
        )
    )

    payload = json.loads(export_store_backup(store))

    assert payload["metadata"] == {
        "format": "graph.store.backup.v1",
        "schema_version": 6,
        "include_embeddings": False,
        "unit_count": 2,
        "edge_count": 2,
        "sync_state_count": 2,
        "embedding_encoding": None,
    }
    assert [unit["id"] for unit in payload["units"]] == ["unit-a", "unit-b"]
    assert payload["units"][0] == {
        "id": "unit-a",
        "source_project": "max",
        "source_id": "a",
        "source_entity_type": "insight",
        "title": "Unit unit-a",
        "content": "Content for unit-a",
        "content_type": "insight",
        "metadata": {"rank": "a", "nested": {"flag": True}},
        "tags": ["backup", "a"],
        "confidence": 0.7,
        "utility_score": 0.9,
        "created_at": "2026-01-01T00:00:00+00:00",
        "ingested_at": payload["units"][0]["ingested_at"],
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    assert [edge["id"] for edge in payload["edges"]] == ["edge-a", "edge-b"]
    assert payload["edges"][0] == {
        "id": "edge-a",
        "from_unit_id": "unit-a",
        "to_unit_id": "unit-b",
        "relation": "builds_on",
        "weight": 0.5,
        "source": "manual",
        "metadata": {"why": "backup"},
        "created_at": "2026-01-03T00:00:00+00:00",
    }
    assert payload["sync_state"] == [
        {
            "source_project": "forty_two",
            "source_entity_type": "knowledge_node",
            "last_sync_at": "2026-01-06T00:00:00+00:00",
            "last_source_id": "node-1",
            "items_synced": 1,
        },
        {
            "source_project": "max",
            "source_entity_type": "insight",
            "last_sync_at": "2026-01-05T00:00:00+00:00",
            "last_source_id": "b",
            "items_synced": 2,
        },
    ]


def test_export_store_backup_is_deterministic_and_writes_returned_json(store: Store, tmp_path):
    store.insert_unit(_unit("unit-a", "a", _dt("2026-01-01T00:00:00+00:00")))
    output_path = tmp_path / "nested" / "backup.json"

    first = export_store_backup(store, path=output_path)
    second = export_store_backup(store)

    assert first == second
    assert output_path.read_text(encoding="utf-8") == first


def test_export_store_backup_omits_embeddings_by_default_and_includes_when_requested(
    store: Store,
):
    unit = store.insert_unit(_unit("unit-a", "a", _dt("2026-01-01T00:00:00+00:00")))
    embedding = b"\x00\x01backup-embedding"
    store.update_embedding(unit.id, embedding)

    without_embeddings = json.loads(export_store_backup(store))
    with_embeddings = json.loads(export_store_backup(store, include_embeddings=True))

    assert "embedding" not in without_embeddings["units"][0]
    assert without_embeddings["metadata"]["include_embeddings"] is False
    assert without_embeddings["metadata"]["embedding_encoding"] is None
    assert with_embeddings["metadata"]["include_embeddings"] is True
    assert with_embeddings["metadata"]["embedding_encoding"] == "base64"
    assert with_embeddings["units"][0]["embedding"] == base64.b64encode(embedding).decode(
        "ascii"
    )
