"""Tests for unit activity summaries."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    for candidate in (
        Path(path),
        Path(path).with_name(Path(path).name + "-wal"),
        Path(path).with_name(Path(path).name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def unit(
    unit_id: str,
    created_at: str,
    *,
    updated_at: str | None = None,
    source_project: SourceProject = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="insight",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=content_type,
        created_at=dt(created_at),
        updated_at=dt(updated_at or created_at),
    )


def set_ingested_at(store: Store, unit_id: str, value: str) -> None:
    store.conn.execute(
        "UPDATE knowledge_units SET ingested_at = ? WHERE id = ?",
        (dt(value).isoformat(), unit_id),
    )
    store.conn.commit()


def test_unit_activity_summary_buckets_months_with_breakdowns(store: Store):
    store.insert_unit(
        unit(
            "alpha",
            "2026-01-15T10:00:00",
            source_project=SourceProject.MAX,
            content_type=ContentType.INSIGHT,
        )
    )
    store.insert_unit(
        unit(
            "beta",
            "2026-01-20T10:00:00",
            source_project=SourceProject.FORTY_TWO,
            content_type=ContentType.FINDING,
        )
    )
    store.insert_unit(
        unit(
            "gamma",
            "2026-02-02T10:00:00",
            source_project=SourceProject.MAX,
            content_type=ContentType.FINDING,
        )
    )

    assert store.unit_activity_summary() == {
        "field": "created_at",
        "bucket": "month",
        "start": None,
        "end": None,
        "include_empty": False,
        "buckets": [
            {
                "bucket": "2026-01",
                "count": 2,
                "source_project_counts": {"forty_two": 1, "max": 1},
                "content_type_counts": {"finding": 1, "insight": 1},
            },
            {
                "bucket": "2026-02",
                "count": 1,
                "source_project_counts": {"max": 1},
                "content_type_counts": {"finding": 1},
            },
        ],
        "first_seen_at": "2026-01-15T10:00:00+00:00",
        "last_seen_at": "2026-02-02T10:00:00+00:00",
    }


def test_unit_activity_summary_filters_start_and_end_inclusively(store: Store):
    store.insert_unit(unit("before", "2026-01-31T23:59:59"))
    store.insert_unit(unit("start", "2026-02-01T00:00:00"))
    store.insert_unit(unit("middle", "2026-02-15T12:00:00"))
    store.insert_unit(unit("end", "2026-02-28T23:59:59"))
    store.insert_unit(unit("after", "2026-03-01T00:00:00"))

    summary = store.unit_activity_summary(
        bucket="day",
        start="2026-02-01T00:00:00+00:00",
        end="2026-02-28T23:59:59+00:00",
    )

    assert [row["bucket"] for row in summary["buckets"]] == [
        "2026-02-01",
        "2026-02-15",
        "2026-02-28",
    ]
    assert [row["count"] for row in summary["buckets"]] == [1, 1, 1]
    assert summary["first_seen_at"] == "2026-02-01T00:00:00+00:00"
    assert summary["last_seen_at"] == "2026-02-28T23:59:59+00:00"


def test_unit_activity_summary_supports_week_and_updated_at(store: Store):
    store.insert_unit(unit("first", "2026-01-01T00:00:00", updated_at="2026-03-04T08:00:00"))
    store.insert_unit(unit("second", "2026-01-02T00:00:00", updated_at="2026-03-08T08:00:00"))
    store.insert_unit(unit("third", "2026-01-03T00:00:00", updated_at="2026-03-09T08:00:00"))

    assert store.unit_activity_summary(field="updated_at", bucket="week")["buckets"] == [
        {
            "bucket": "2026-03-02",
            "count": 2,
            "source_project_counts": {"max": 2},
            "content_type_counts": {"insight": 2},
        },
        {
            "bucket": "2026-03-09",
            "count": 1,
            "source_project_counts": {"max": 1},
            "content_type_counts": {"insight": 1},
        },
    ]


def test_unit_activity_summary_supports_ingested_at(store: Store):
    store.insert_unit(unit("first", "2026-01-01T00:00:00"))
    store.insert_unit(unit("second", "2026-01-02T00:00:00"))
    set_ingested_at(store, "first", "2026-04-01T10:00:00")
    set_ingested_at(store, "second", "2026-04-02T10:00:00")

    assert store.unit_activity_summary(field="ingested_at", bucket="day")["buckets"] == [
        {
            "bucket": "2026-04-01",
            "count": 1,
            "source_project_counts": {"max": 1},
            "content_type_counts": {"insight": 1},
        },
        {
            "bucket": "2026-04-02",
            "count": 1,
            "source_project_counts": {"max": 1},
            "content_type_counts": {"insight": 1},
        },
    ]


def test_unit_activity_summary_include_empty_fills_gaps_between_start_and_end(
    store: Store,
):
    store.insert_unit(unit("first", "2026-01-15T10:00:00"))
    store.insert_unit(unit("second", "2026-03-01T10:00:00"))

    summary = store.unit_activity_summary(
        bucket="month",
        start="2026-01-01",
        end="2026-03-31T23:59:59+00:00",
        include_empty=True,
    )

    assert summary["buckets"] == [
        {
            "bucket": "2026-01",
            "count": 1,
            "source_project_counts": {"max": 1},
            "content_type_counts": {"insight": 1},
        },
        {
            "bucket": "2026-02",
            "count": 0,
            "source_project_counts": {},
            "content_type_counts": {},
        },
        {
            "bucket": "2026-03",
            "count": 1,
            "source_project_counts": {"max": 1},
            "content_type_counts": {"insight": 1},
        },
    ]


@pytest.mark.parametrize("field", ["", "published_at", "title"])
def test_unit_activity_summary_rejects_invalid_fields(store: Store, field: str):
    with pytest.raises(ValueError, match="field"):
        store.unit_activity_summary(field=field)


@pytest.mark.parametrize("bucket", ["", "year", "hour"])
def test_unit_activity_summary_rejects_invalid_buckets(store: Store, bucket: str):
    with pytest.raises(ValueError, match="bucket"):
        store.unit_activity_summary(bucket=bucket)


def test_unit_activity_summary_rejects_invalid_date_ranges(store: Store):
    with pytest.raises(ValueError, match="start"):
        store.unit_activity_summary(start="not-a-date")

    with pytest.raises(ValueError, match="end"):
        store.unit_activity_summary(end="not-a-date")

    with pytest.raises(ValueError, match="start must be on or before end"):
        store.unit_activity_summary(start="2026-02-01", end="2026-01-01")
