"""Tests for graph activity burst analysis."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from types import SimpleNamespace

import pytest

from graph.graph.service import GraphService
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
    os.unlink(path)


def _unit(
    unit_id: str,
    title: str,
    created_at: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
    tags: list[str] | None = None,
    content_type: ContentType = ContentType.INSIGHT,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="insight",
        title=title,
        content=f"{title} activity note",
        content_type=content_type,
        tags=tags or [],
        created_at=datetime.fromisoformat(created_at),
    )


def test_analyze_activity_bursts_returns_month_bursts_with_distributions(store: Store):
    for unit in [
        _unit("jan-1", "January baseline", "2026-01-15T10:00:00+00:00"),
        _unit("feb-2", "February B", "2026-02-05T10:00:00+00:00", tags=["grid"]),
        _unit("feb-1", "February A", "2026-02-01T10:00:00+00:00", tags=["solar"]),
        _unit(
            "feb-3",
            "February C",
            "2026-02-20T10:00:00+00:00",
            source_project=SourceProject.PRESENCE,
            tags=["solar", "grid"],
            content_type=ContentType.ARTIFACT,
        ),
        _unit("feb-4", "February D", "2026-02-25T10:00:00+00:00", tags=["solar"]),
        _unit("mar-1", "March baseline", "2026-03-10T10:00:00+00:00"),
        _unit("may-1", "May A", "2026-05-01T10:00:00+00:00", tags=["archive"]),
        _unit("may-2", "May B", "2026-05-02T10:00:00+00:00", tags=["archive"]),
        _unit("may-3", "May C", "2026-05-03T10:00:00+00:00", tags=["archive"]),
    ]:
        store.insert_unit(unit)

    result = GraphService(store).analyze_activity_bursts(
        bucket="month",
        min_count=2,
        limit=1,
        examples_per_burst=2,
    )

    assert result["bucket"] == "month"
    assert result["total"] == 9
    assert result["dated_count"] == 9
    assert result["undated_count"] == 0
    assert result["burst_count"] == 2
    assert [burst["period"] for burst in result["bursts"]] == ["2026-02"]

    february = result["bursts"][0]
    assert february["count"] == 4
    assert february["local_baseline"] == 1
    assert february["source_distribution"] == {"max": 3, "presence": 1}
    assert february["tag_distribution"] == {"solar": 3, "grid": 2}
    assert [unit["id"] for unit in february["example_units"]] == ["feb-1", "feb-2"]


def test_analyze_activity_bursts_supports_day_buckets(store: Store):
    for unit in [
        _unit("apr-09", "April 9", "2026-04-09T10:00:00+00:00"),
        _unit("apr-10-a", "April 10 A", "2026-04-10T09:00:00+00:00"),
        _unit("apr-10-b", "April 10 B", "2026-04-10T11:00:00+00:00"),
        _unit("apr-10-c", "April 10 C", "2026-04-10T12:00:00+00:00"),
        _unit("apr-11", "April 11", "2026-04-11T10:00:00+00:00"),
    ]:
        store.insert_unit(unit)

    result = GraphService(store).analyze_activity_bursts(bucket="day")

    assert result["burst_count"] == 1
    assert result["bursts"][0]["period"] == "2026-04-10"
    assert result["bursts"][0]["count"] == 3
    assert result["bursts"][0]["local_baseline"] == 1


def test_analyze_activity_bursts_counts_undated_units():
    dated = SimpleNamespace(
        id="dated",
        title="Dated",
        source_project="max",
        source_id="dated",
        content_type="insight",
        tags=["solar"],
        created_at="2026-06-01T10:00:00+00:00",
        ingested_at=None,
        updated_at=None,
    )
    undated = SimpleNamespace(
        id="undated",
        title="Undated",
        source_project="presence",
        source_id="undated",
        content_type="artifact",
        tags=["draft"],
        created_at=None,
        ingested_at="",
        updated_at="not-a-date",
    )
    fake_store = SimpleNamespace(get_all_units=lambda limit: [dated, undated])

    result = GraphService(fake_store).analyze_activity_bursts(min_count=1)

    assert result["total"] == 2
    assert result["dated_count"] == 1
    assert result["undated_count"] == 1
    assert result["bursts"][0]["period"] == "2026-06"


@pytest.mark.parametrize(
    ("arguments", "error_type", "message"),
    [
        ({"bucket": "week"}, ValueError, "Unsupported activity burst bucket"),
        ({"min_count": -1}, ValueError, "min_count must be a non-negative integer"),
        ({"limit": True}, TypeError, "limit must be a non-negative integer"),
        (
            {"examples_per_burst": "many"},
            TypeError,
            "examples_per_burst must be a non-negative integer",
        ),
    ],
)
def test_analyze_activity_bursts_validates_arguments(
    store: Store,
    arguments,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        GraphService(store).analyze_activity_bursts(**arguments)
