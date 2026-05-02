from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


def _insert_unit(
    store: Store,
    unit_id: str,
    title: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    updated_at: datetime | None = None,
):
    timestamp = updated_at or datetime(2026, 4, 20, tzinfo=timezone.utc)
    return store.insert_unit(
        KnowledgeUnit(
            id=unit_id,
            source_project=source_project,
            source_id=unit_id,
            source_entity_type="insight",
            title=title,
            content=f"{title} content",
            content_type=ContentType.INSIGHT,
            created_at=timestamp,
            ingested_at=timestamp,
            updated_at=timestamp,
        )
    )


def _insert_edge(store: Store, from_unit_id: str, to_unit_id: str):
    return store.insert_edge(
        KnowledgeEdge(
            from_unit_id=from_unit_id,
            to_unit_id=to_unit_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.MANUAL,
        )
    )


@pytest.fixture
def isolates_store(store: Store):
    connected_a = _insert_unit(
        store,
        "connected-a",
        "Connected A",
        source_project=SourceProject.FORTY_TWO,
    )
    connected_b = _insert_unit(
        store,
        "connected-b",
        "Connected B",
        source_project=SourceProject.MAX,
    )
    _insert_unit(
        store,
        "isolated-old",
        "Old Isolate",
        source_project=SourceProject.MAX,
        updated_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
    )
    _insert_unit(
        store,
        "isolated-beta",
        "Beta Isolate",
        source_project=SourceProject.PRESENCE,
        updated_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
    )
    _insert_unit(
        store,
        "isolated-alpha",
        "Alpha Isolate",
        source_project=SourceProject.MAX,
        updated_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
    )
    _insert_edge(store, connected_a.id, connected_b.id)
    return store


def test_isolated_units_empty_graph_returns_zero_counts(store: Store):
    assert GraphService(store).isolated_units() == {
        "isolated_count": 0,
        "total_units": 0,
        "ratio": 0.0,
        "filters": {
            "source_project": None,
            "limit": 50,
            "include_units": True,
        },
        "units": [],
    }


def test_isolated_units_reports_count_ratio_and_sorted_units(isolates_store: Store):
    result = GraphService(isolates_store).isolated_units(limit=10)

    assert result["isolated_count"] == 3
    assert result["total_units"] == 5
    assert result["ratio"] == pytest.approx(0.6)
    assert [unit["id"] for unit in result["units"]] == [
        "isolated-alpha",
        "isolated-beta",
        "isolated-old",
    ]
    assert result["units"][0]["title"] == "Alpha Isolate"
    assert result["units"][0]["source_project"] == "max"


def test_isolated_units_source_filter_does_not_change_total_graph_state(
    isolates_store: Store,
):
    result = GraphService(isolates_store).isolated_units(
        source_project=SourceProject.MAX,
        limit=10,
    )

    assert result["isolated_count"] == 2
    assert result["total_units"] == 5
    assert result["ratio"] == pytest.approx(0.4)
    assert result["filters"] == {
        "source_project": "max",
        "limit": 10,
        "include_units": True,
    }
    assert [unit["id"] for unit in result["units"]] == [
        "isolated-alpha",
        "isolated-old",
    ]


def test_isolated_units_accepts_zero_limit(isolates_store: Store):
    result = GraphService(isolates_store).isolated_units(limit=0)

    assert result["isolated_count"] == 3
    assert result["total_units"] == 5
    assert result["units"] == []
    assert result["filters"]["limit"] == 0


def test_isolated_units_can_omit_unit_payloads(isolates_store: Store):
    result = GraphService(isolates_store).isolated_units(include_units=False)

    assert result["isolated_count"] == 3
    assert result["total_units"] == 5
    assert result["units"] == []
    assert result["filters"]["include_units"] is False


@pytest.mark.parametrize("limit", [-1, "many", None, True])
def test_isolated_units_rejects_invalid_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        GraphService(store).isolated_units(limit=limit)
