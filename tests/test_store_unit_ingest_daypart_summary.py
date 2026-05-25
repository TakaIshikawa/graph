from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.store.db import Store
from graph.store.unit_ingest_daypart_summary import summarize_unit_ingest_daypart
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject,
    content_type: ContentType,
    created_at: str,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="page",
        title=unit_id,
        content=f"Content {unit_id}",
        content_type=content_type,
        created_at=dt(created_at),
    )


def set_ingested_at(store: Store, unit_id: str, value: str | None) -> None:
    store.conn.execute("UPDATE knowledge_units SET ingested_at = ? WHERE id = ?", (value, unit_id))
    store.conn.commit()


def test_summarize_unit_ingest_daypart_applies_timezone_offset_and_counts(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.insert_unit(
            unit(
                "morning",
                source_project=SourceProject.MAX,
                content_type=ContentType.INSIGHT,
                created_at="2026-01-04T00:00:00",
            )
        )
        store.insert_unit(
            unit(
                "morning-two",
                source_project=SourceProject.CSV,
                content_type=ContentType.FINDING,
                created_at="2026-01-04T00:00:00",
            )
        )
        store.insert_unit(
            unit(
                "evening",
                source_project=SourceProject.MAX,
                content_type=ContentType.ARTIFACT,
                created_at="2026-01-04T00:00:00",
            )
        )
        set_ingested_at(store, "morning", "2026-01-04T21:30:00+00:00")
        set_ingested_at(store, "morning-two", "2026-01-04T21:45:00+00:00")
        set_ingested_at(store, "evening", "2026-01-05T09:00:00+00:00")

        summary = summarize_unit_ingest_daypart(store, timezone_offset_hours=9)
    finally:
        store.close()

    assert summary["rows"] == [
        {
            "weekday": "monday",
            "hour": 6,
            "daypart": "morning",
            "unit_count": 2,
            "source_projects": ["csv", "max"],
            "content_type_counts": {"finding": 1, "insight": 1},
        },
        {
            "weekday": "monday",
            "hour": 18,
            "daypart": "evening",
            "unit_count": 1,
            "source_projects": ["max"],
            "content_type_counts": {"artifact": 1},
        },
    ]
    assert summary["skipped_rows"] == []


def test_summarize_unit_ingest_daypart_falls_back_and_skips_missing_timestamps(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.insert_unit(
            unit(
                "fallback",
                source_project=SourceProject.MAX,
                content_type=ContentType.INSIGHT,
                created_at="2026-01-05T13:00:00",
            )
        )
        set_ingested_at(store, "fallback", "not-a-date")

        summary = summarize_unit_ingest_daypart(store)
    finally:
        store.close()

    assert summary["rows"][0]["daypart"] == "afternoon"
    assert summary["rows"][0]["hour"] == 13


def test_summarize_unit_ingest_daypart_reports_missing_nullable_timestamp(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.insert_unit(
            unit(
                "missing",
                source_project=SourceProject.MAX,
                content_type=ContentType.INSIGHT,
                created_at="2026-01-05T00:00:00",
            )
        )

        summary = summarize_unit_ingest_daypart(store, timestamp_fields=["embedding_updated_at"])
    finally:
        store.close()

    assert summary["rows"] == []
    assert summary["skipped_rows"] == [{"unit_id": "missing", "reason": "missing_timestamp"}]


def test_summarize_unit_ingest_daypart_validates_timestamp_fields(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        with pytest.raises(ValueError, match="unsupported fields"):
            summarize_unit_ingest_daypart(store, timestamp_fields=["metadata.created_at"])
    finally:
        store.close()
