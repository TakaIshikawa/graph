from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


AS_OF = datetime(2026, 5, 1, tzinfo=timezone.utc)


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
        created_at=AS_OF,
        updated_at=AS_OF,
    )


def _set_dates(
    store: Store,
    unit_id: str,
    *,
    created_at: str,
    updated_at: str,
    ingested_at: str,
) -> None:
    store.conn.execute(
        """UPDATE knowledge_units
           SET created_at = ?, updated_at = ?, ingested_at = ?
           WHERE id = ?""",
        (created_at, updated_at, ingested_at, unit_id),
    )
    store.conn.commit()


def test_freshness_summary_counts_default_age_buckets(store: Store):
    rows = [
        ("unit-7d", "Fresh", "2026-04-30T00:00:00+00:00"),
        ("unit-30d", "Recent", "2026-04-11T00:00:00+00:00"),
        ("unit-90d", "Aging", "2026-03-02T00:00:00+00:00"),
        ("unit-older", "Old", "2026-01-01T00:00:00+00:00"),
        ("unit-invalid", "Invalid", "not-a-date"),
        ("unit-missing", "Missing", ""),
    ]
    for unit_id, title, updated_at in rows:
        store.insert_unit(_unit(unit_id, title))
        _set_dates(
            store,
            unit_id,
            created_at="2026-04-30T00:00:00+00:00",
            updated_at=updated_at,
            ingested_at="2026-04-30T00:00:00+00:00",
        )

    result = GraphService(store).freshness_summary(as_of=AS_OF)

    assert result == {
        "field": "updated_at",
        "as_of": "2026-05-01T00:00:00+00:00",
        "total": 6,
        "dated_count": 4,
        "undated_count": 2,
        "buckets": [
            {"bucket": "7d", "min_age_days": None, "max_age_days": 7, "count": 1},
            {"bucket": "30d", "min_age_days": 7, "max_age_days": 30, "count": 1},
            {"bucket": "90d", "min_age_days": 30, "max_age_days": 90, "count": 1},
            {"bucket": "older", "min_age_days": 90, "max_age_days": None, "count": 1},
        ],
    }


def test_freshness_summary_supports_created_updated_and_ingested_fields(store: Store):
    store.insert_unit(_unit("unit-alpha", "Alpha"))
    store.insert_unit(_unit("unit-beta", "Beta"))
    _set_dates(
        store,
        "unit-alpha",
        created_at="2026-04-30T00:00:00+00:00",
        updated_at="2026-03-15T00:00:00+00:00",
        ingested_at="2026-01-01T00:00:00+00:00",
    )
    _set_dates(
        store,
        "unit-beta",
        created_at="2026-02-15T00:00:00+00:00",
        updated_at="2026-04-30T00:00:00+00:00",
        ingested_at="2026-04-15T00:00:00+00:00",
    )

    service = GraphService(store)

    created_buckets = service.freshness_summary("created_at", as_of=AS_OF)["buckets"]
    updated_buckets = service.freshness_summary("updated_at", as_of=AS_OF)["buckets"]
    ingested_buckets = service.freshness_summary("ingested_at", as_of=AS_OF)["buckets"]

    assert [item["count"] for item in created_buckets] == [1, 0, 1, 0]
    assert [item["count"] for item in updated_buckets] == [1, 0, 1, 0]
    assert [item["count"] for item in ingested_buckets] == [0, 1, 0, 1]


def test_freshness_summary_custom_buckets_are_deterministic(store: Store):
    for unit_id, updated_at in [
        ("unit-a", "2026-04-30T00:00:00+00:00"),
        ("unit-b", "2026-04-20T00:00:00+00:00"),
        ("unit-c", "2026-03-15T00:00:00+00:00"),
    ]:
        store.insert_unit(_unit(unit_id, unit_id))
        _set_dates(
            store,
            unit_id,
            created_at=updated_at,
            updated_at=updated_at,
            ingested_at=updated_at,
        )

    result = GraphService(store).freshness_summary(
        buckets=["14d", "older"],
        as_of="2026-05-01T00:00:00Z",
    )

    assert result["as_of"] == "2026-05-01T00:00:00+00:00"
    assert result["buckets"] == [
        {"bucket": "14d", "min_age_days": None, "max_age_days": 14, "count": 2},
        {"bucket": "older", "min_age_days": 14, "max_age_days": None, "count": 1},
    ]


def test_freshness_summary_validates_field_buckets_and_as_of(store: Store):
    service = GraphService(store)

    with pytest.raises(ValueError, match="Unsupported freshness field"):
        service.freshness_summary("deleted_at")

    with pytest.raises(ValueError, match="positive values like '7d'"):
        service.freshness_summary(buckets=["7"])

    with pytest.raises(ValueError, match="strictly increasing"):
        service.freshness_summary(buckets=["30d", "7d"])

    with pytest.raises(ValueError, match="older.*end"):
        service.freshness_summary(buckets=["older", "7d"])

    with pytest.raises(ValueError, match="as_of must be an ISO-8601"):
        service.freshness_summary(as_of="yesterday")
