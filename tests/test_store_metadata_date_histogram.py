from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def _unit(unit_id: str, metadata: dict) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="event",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def test_metadata_date_histogram_buckets_by_month_with_missing_and_invalid(store: Store):
    for unit in [
        _unit("a", {"event": {"date": "2026-05-01"}}),
        _unit("b", {"event": {"date": "2026-05-20T10:30:00Z"}}),
        _unit("c", {"event": {"date": "2026-06-01"}}),
        _unit("d", {"event": {"date": "not a date"}}),
        _unit("e", {"event": {}}),
    ]:
        store.insert_unit(unit)

    result = store.metadata_date_histogram("event.date")

    assert result == {
        "path": "event.date",
        "bucket": "month",
        "unit_count": 5,
        "valid_count": 3,
        "missing_count": 1,
        "invalid_count": 1,
        "buckets": [
            {"bucket": "2026-05", "count": 2},
            {"bucket": "2026-06", "count": 1},
        ],
    }


def test_metadata_date_histogram_supports_day_year_and_limit(store: Store):
    for unit in [
        _unit("a", {"date": "2025-12-31"}),
        _unit("b", {"date": "2026-01-01"}),
        _unit("c", {"date": "2026-01-02"}),
    ]:
        store.insert_unit(unit)

    day = store.metadata_date_histogram("date", bucket="day", limit=2)
    year = store.metadata_date_histogram("date", bucket="year")

    assert day["buckets"] == [
        {"bucket": "2025-12-31", "count": 1},
        {"bucket": "2026-01-01", "count": 1},
    ]
    assert year["buckets"] == [
        {"bucket": "2025", "count": 1},
        {"bucket": "2026", "count": 2},
    ]


def test_metadata_date_histogram_rejects_invalid_bucket(store: Store):
    with pytest.raises(ValueError, match="Unsupported metadata date histogram bucket"):
        store.metadata_date_histogram("date", bucket="week")
