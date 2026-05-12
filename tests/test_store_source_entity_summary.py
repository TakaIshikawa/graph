from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def _dt(day: int) -> datetime:
    return datetime(2026, 5, day, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def _unit(
    unit_id: str,
    source_project: SourceProject,
    entity_type: str,
    *,
    created_at: datetime,
    updated_at: datetime,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type=entity_type,
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=["sample"],
        created_at=created_at,
        updated_at=updated_at,
    )


def test_source_entity_type_summary_groups_dates_and_samples(store: Store):
    store.insert_unit(_unit("max-1", SourceProject.MAX, "note", created_at=_dt(1), updated_at=_dt(4)))
    store.insert_unit(_unit("max-2", SourceProject.MAX, "note", created_at=_dt(2), updated_at=_dt(5)))
    store.insert_unit(_unit("presence-1", SourceProject.PRESENCE, "", created_at=_dt(3), updated_at=_dt(3)))

    result = store.source_entity_type_summary(sample_limit=1)

    assert [(row["source_project"], row["source_entity_type"], row["count"]) for row in result] == [
        ("max", "note", 2),
        ("presence", "", 1),
    ]
    assert result[0]["earliest_created_at"] == "2026-05-01T00:00:00+00:00"
    assert result[0]["latest_created_at"] == "2026-05-02T00:00:00+00:00"
    assert result[0]["latest_updated_at"] == "2026-05-05T00:00:00+00:00"
    assert result[0]["sample_units"] == [
        {
            "id": "max-1",
            "title": "Unit max-1",
            "source_project": "max",
            "source_id": "max-1",
            "source_entity_type": "note",
            "content_type": "insight",
            "tags": ["sample"],
        }
    ]


def test_source_entity_type_summary_orders_ties_deterministically(store: Store):
    store.insert_unit(_unit("b", SourceProject.PRESENCE, "entry", created_at=_dt(1), updated_at=_dt(1)))
    store.insert_unit(_unit("a", SourceProject.MAX, "entry", created_at=_dt(1), updated_at=_dt(1)))

    result = store.source_entity_type_summary()

    assert [(row["source_project"], row["source_entity_type"]) for row in result] == [
        ("max", "entry"),
        ("presence", "entry"),
    ]


def test_source_entity_type_summary_rejects_invalid_sample_limit(store: Store):
    with pytest.raises(ValueError, match="sample_limit must be a non-negative integer"):
        store.source_entity_type_summary(sample_limit=-1)
