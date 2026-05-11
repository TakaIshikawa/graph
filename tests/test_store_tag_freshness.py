from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from graph.store.db import Store
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "graph.db"
    s = Store(str(path))
    yield s
    s.close()
    for candidate in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        candidate.unlink(missing_ok=True)


def _unit(unit_id: str, title: str, *, tags: list[str], created_at: datetime, source_project=SourceProject.MAX):
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=title,
        tags=tags,
        created_at=created_at,
        updated_at=created_at,
    )


def test_tag_freshness_summary_counts_stale_units_and_sources(store: Store):
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=120)
    fresh = now - timedelta(days=10)
    store.insert_unit(_unit("old", "Old Solar", tags=["solar", "grid"], created_at=old, source_project=SourceProject.CSV))
    store.insert_unit(_unit("fresh", "Fresh Solar", tags=["solar"], created_at=fresh))

    result = store.tag_freshness_summary(stale_after_days=90)

    assert result[0]["tag"] == "solar"
    assert result[0]["unit_count"] == 2
    assert result[0]["stale_unit_count"] == 1
    assert result[0]["source_projects"] == ["csv", "max"]
    assert result[0]["example_titles"] == ["Fresh Solar", "Old Solar"]
    assert result[1]["tag"] == "grid"
    assert result[1]["stale_unit_count"] == 1


def test_tag_freshness_summary_limit_and_empty_store(store: Store):
    now = datetime.now(timezone.utc) - timedelta(days=1)
    store.insert_unit(_unit("a", "A", tags=["b", "a"], created_at=now))

    assert [row["tag"] for row in store.tag_freshness_summary(limit=1)] == ["a"]

    empty = Store(str(store.db_path.with_name("empty.db")))
    try:
        assert empty.tag_freshness_summary() == []
    finally:
        empty.close()


@pytest.mark.parametrize("kwargs", [{"limit": -1}, {"stale_after_days": -1}, {"limit": True}])
def test_tag_freshness_summary_validates_arguments(store: Store, kwargs):
    with pytest.raises(ValueError):
        store.tag_freshness_summary(**kwargs)
