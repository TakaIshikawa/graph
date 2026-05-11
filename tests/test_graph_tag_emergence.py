from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _unit(unit_id: str, title: str, tags: list[str], created_at: str) -> KnowledgeUnit:
    timestamp = _dt(created_at)
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        tags=tags,
        created_at=timestamp,
        ingested_at=timestamp,
        updated_at=timestamp,
    )


def test_analyze_tag_emergence_identifies_emerging_and_fading_tags(store: Store):
    for item in [
        _unit("old-a", "Old A", ["legacy"], "2026-01-10T00:00:00"),
        _unit("old-b", "Old B", ["legacy"], "2026-02-10T00:00:00"),
        _unit("new-a", "New A", ["rag"], "2026-03-10T00:00:00"),
        _unit("new-b", "New B", ["rag"], "2026-04-10T00:00:00"),
        _unit("new-c", "New C", ["rag"], "2026-04-11T00:00:00"),
    ]:
        store.insert_unit(item)

    result = GraphService(store).analyze_tag_emergence(recent_buckets=2)

    assert result["bucket_count"] == 4
    assert result["tag_count"] == 2
    assert result["options"] == {
        "bucket": "month",
        "field": "created_at",
        "recent_buckets": 2,
        "limit": 20,
    }
    assert result["emerging_tags"][0]["tag"] == "rag"
    assert result["emerging_tags"][0]["bucket_counts"] == {"2026-03": 1, "2026-04": 2}
    assert result["emerging_tags"][0]["recent_count"] == 3
    assert result["emerging_tags"][0]["previous_count"] == 0
    assert result["emerging_tags"][0]["representative_units"][0]["id"] == "new-b"
    assert result["fading_tags"][0]["tag"] == "legacy"


def test_analyze_tag_emergence_supports_updated_at_field(store: Store):
    item = _unit("a", "Alpha", ["alpha"], "2026-01-10T00:00:00")
    item.updated_at = _dt("2026-05-10T00:00:00")
    store.insert_unit(item)

    result = GraphService(store).analyze_tag_emergence(field="updated_at", bucket="day")

    assert result["buckets"] == ["2026-05-10"]
    assert result["emerging_tags"][0]["first_seen"] == "2026-05-10"


@pytest.mark.parametrize("field", ["published_at", "", None])
def test_analyze_tag_emergence_validates_field(store: Store, field):
    with pytest.raises(ValueError, match="Unsupported tag emergence field"):
        GraphService(store).analyze_tag_emergence(field=field)
