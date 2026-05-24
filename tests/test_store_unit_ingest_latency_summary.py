from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.store.db import Store
from graph.store.unit_ingest_latency_summary import summarize_unit_ingest_latency
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    yield store
    store.close()


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def unit(unit_id: str, *, fetched_at: str | None, created_at: str) -> KnowledgeUnit:
    metadata = {}
    if fetched_at is not None:
        metadata["fetched_at"] = fetched_at
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="page",
        title=unit_id,
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
        created_at=dt(created_at),
    )


def set_ingested_at(store: Store, unit_id: str, value: str) -> None:
    store.conn.execute(
        "UPDATE knowledge_units SET ingested_at = ? WHERE id = ?",
        (dt(value).isoformat(), unit_id),
    )
    store.conn.commit()


def test_summarize_unit_ingest_latency_reports_aggregates_and_buckets(store: Store):
    store.insert_unit(unit("fast", fetched_at="2026-01-01T00:00:00", created_at="2026-01-01T00:00:01"))
    store.insert_unit(unit("slow", fetched_at="2026-01-01T00:00:00", created_at="2026-01-01T00:00:01"))
    set_ingested_at(store, "fast", "2026-01-01T00:01:00")
    set_ingested_at(store, "slow", "2026-01-01T00:10:00")

    summary = summarize_unit_ingest_latency(store, bucket_bounds_seconds=[60, 600])

    assert summary["count"] == 2
    assert summary["min_seconds"] == 60.0
    assert summary["max_seconds"] == 600.0
    assert summary["average_seconds"] == 330.0
    assert summary["buckets"] == [
        {"bucket": "0-60", "count": 1},
        {"bucket": "60-600", "count": 1},
        {"bucket": ">600", "count": 0},
    ]
    assert [row["bucket"] for row in summary["latency_rows"]] == ["0-60", "60-600"]
    assert summary["skipped_rows"] == []


def test_summarize_unit_ingest_latency_skips_missing_and_negative_rows(store: Store):
    store.insert_unit(unit("missing-source", fetched_at=None, created_at="2026-01-01T00:00:00"))
    store.insert_unit(
        unit("negative", fetched_at="2026-01-01T00:10:00", created_at="2026-01-01T00:00:00")
    )
    set_ingested_at(store, "missing-source", "2026-01-01T00:01:00")
    set_ingested_at(store, "negative", "2026-01-01T00:05:00")

    summary = summarize_unit_ingest_latency(store)

    assert summary["count"] == 0
    assert summary["min_seconds"] is None
    assert summary["max_seconds"] is None
    assert summary["average_seconds"] is None
    assert summary["skipped_count"] == 2
    assert {row["reason"] for row in summary["skipped_rows"]} == {
        "missing_source_timestamp",
        "negative_latency",
    }


def test_summarize_unit_ingest_latency_can_use_created_at_and_custom_source_keys(store: Store):
    inserted = unit("created", fetched_at=None, created_at="2026-01-01T00:02:00")
    inserted.metadata = {"source": {"imported_at": "2026-01-01T00:00:00Z"}}
    store.insert_unit(inserted)

    summary = summarize_unit_ingest_latency(
        store,
        source_timestamp_keys=["source.imported_at"],
        unit_timestamp_fields=["created_at"],
        bucket_bounds_seconds=[120],
    )

    assert summary["count"] == 1
    assert summary["latency_rows"][0]["latency_seconds"] == 120.0
    assert summary["latency_rows"][0]["source_timestamp_key"] == "source.imported_at"
    assert summary["latency_rows"][0]["unit_timestamp_field"] == "created_at"
    assert summary["buckets"] == [
        {"bucket": "0-120", "count": 1},
        {"bucket": ">120", "count": 0},
    ]


def test_summarize_unit_ingest_latency_validates_bucket_boundaries(store: Store):
    with pytest.raises(ValueError, match="strictly increasing"):
        summarize_unit_ingest_latency(store, bucket_bounds_seconds=[60, 60])
