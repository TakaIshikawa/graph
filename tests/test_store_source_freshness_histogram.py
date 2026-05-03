"""Tests for store source freshness histograms."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
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


def unit(
    unit_id: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
    updated_at: datetime | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="insight",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        updated_at=updated_at or datetime.now(timezone.utc),
    )


def test_source_freshness_histogram_counts_mixed_timestamps_by_source(store: Store):
    now = datetime.now(timezone.utc)
    store.insert_unit(unit("max-fresh", updated_at=now - timedelta(days=5)))
    store.insert_unit(unit("max-aging", updated_at=now - timedelta(days=45)))
    store.insert_unit(unit("max-stale", updated_at=now - timedelta(days=120)))
    store.insert_unit(
        unit(
            "forty-two-fresh",
            source_project=SourceProject.FORTY_TWO,
            updated_at=now - timedelta(days=2),
        )
    )
    store.insert_unit(
        unit(
            "forty-two-stale",
            source_project=SourceProject.FORTY_TWO,
            updated_at=now - timedelta(days=100),
        )
    )

    assert store.source_freshness_histogram() == [
        {
            "source_project": "forty_two",
            "total": 2,
            "fresh": 1,
            "aging": 0,
            "stale": 1,
            "unknown": 0,
        },
        {
            "source_project": "max",
            "total": 3,
            "fresh": 1,
            "aging": 1,
            "stale": 1,
            "unknown": 0,
        },
    ]


def test_source_freshness_histogram_counts_missing_updated_at_as_unknown(store: Store):
    store.insert_unit(unit("max-known"))
    store.insert_unit(unit("max-missing"))
    store.conn.execute(
        "UPDATE knowledge_units SET updated_at = '' WHERE id = ?",
        ("max-missing",),
    )
    store.conn.commit()

    assert store.source_freshness_histogram() == [
        {
            "source_project": "max",
            "total": 2,
            "fresh": 1,
            "aging": 0,
            "stale": 0,
            "unknown": 1,
        }
    ]


def test_source_freshness_histogram_uses_custom_thresholds(store: Store):
    now = datetime.now(timezone.utc)
    store.insert_unit(unit("max-fresh", updated_at=now - timedelta(days=2)))
    store.insert_unit(unit("max-aging", updated_at=now - timedelta(days=10)))
    store.insert_unit(unit("max-stale", updated_at=now - timedelta(days=20)))

    assert store.source_freshness_histogram(fresh_days=3, stale_days=14) == [
        {
            "source_project": "max",
            "total": 3,
            "fresh": 1,
            "aging": 1,
            "stale": 1,
            "unknown": 0,
        }
    ]


def test_source_freshness_histogram_empty_store_returns_empty_rows(store: Store):
    assert store.source_freshness_histogram() == []


@pytest.mark.parametrize(
    ("fresh_days", "stale_days", "match"),
    [
        (-1, 90, "non-negative integers"),
        (30, -1, "non-negative integers"),
        (True, 90, "non-negative integers"),
        (30, False, "non-negative integers"),
        (30.5, 90, "non-negative integers"),
        (91, 90, "less than or equal"),
    ],
)
def test_source_freshness_histogram_rejects_invalid_thresholds(
    store: Store,
    fresh_days: object,
    stale_days: object,
    match: str,
):
    with pytest.raises(ValueError, match=match):
        store.source_freshness_histogram(
            fresh_days=fresh_days,
            stale_days=stale_days,
        )
