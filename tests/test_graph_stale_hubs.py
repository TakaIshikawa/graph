from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    yield store
    store.close()


def _unit(
    unit_id: str,
    title: str,
    *,
    updated_at: datetime,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="insight",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
        metadata=metadata or {},
        updated_at=updated_at,
    )


def _edge(store: Store, from_id: str, to_id: str) -> None:
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.MANUAL,
        )
    )


def test_stale_hubs_filters_by_cutoff_and_degree_with_counts(store: Store):
    for unit in [
        _unit("old-hub", "Old Hub", updated_at=_dt(2026, 1, 1), tags=["review"]),
        _unit("fresh-hub", "Fresh Hub", updated_at=_dt(2026, 5, 1)),
        _unit("old-low", "Old Low", updated_at=_dt(2026, 1, 1)),
        _unit("leaf-a", "Leaf A", updated_at=_dt(2026, 5, 1)),
        _unit("leaf-b", "Leaf B", updated_at=_dt(2026, 5, 1)),
        _unit("leaf-c", "Leaf C", updated_at=_dt(2026, 5, 1)),
        _unit("leaf-d", "Leaf D", updated_at=_dt(2026, 5, 1)),
    ]:
        store.insert_unit(unit)
    _edge(store, "old-hub", "leaf-a")
    _edge(store, "leaf-b", "old-hub")
    _edge(store, "old-hub", "leaf-c")
    _edge(store, "fresh-hub", "leaf-a")
    _edge(store, "leaf-b", "fresh-hub")
    _edge(store, "old-low", "leaf-d")

    result = GraphService(store).stale_hubs(
        _dt(2026, 4, 1),
        min_degree=2,
        now=_dt(2026, 5, 11),
    )

    assert [row["unit"]["id"] for row in result] == ["old-hub"]
    assert result[0]["degree"] == 3
    assert result[0]["inbound_count"] == 1
    assert result[0]["outbound_count"] == 2
    assert result[0]["age_days"] == 130
    assert result[0]["tags"] == ["review"]
    assert result[0]["source"] == {
        "project": "max",
        "id": "old-hub",
        "entity_type": "insight",
    }


def test_stale_hubs_uses_metadata_timestamp_and_orders_stably(store: Store):
    for unit in [
        _unit(
            "hub-b",
            "Same Title",
            updated_at=_dt(2026, 5, 1),
            metadata={"review": {"timestamp": "2026-01-01T00:00:00Z"}},
        ),
        _unit(
            "hub-a",
            "Same Title",
            updated_at=_dt(2026, 5, 1),
            metadata={"review": {"timestamp": "2026-01-01T00:00:00Z"}},
        ),
        _unit("leaf-a", "Leaf A", updated_at=_dt(2026, 5, 1)),
        _unit("leaf-b", "Leaf B", updated_at=_dt(2026, 5, 1)),
        _unit("leaf-c", "Leaf C", updated_at=_dt(2026, 5, 1)),
    ]:
        store.insert_unit(unit)
    _edge(store, "hub-a", "leaf-a")
    _edge(store, "leaf-b", "hub-a")
    _edge(store, "hub-b", "leaf-a")
    _edge(store, "leaf-c", "hub-b")

    result = GraphService(store).stale_hubs(
        _dt(2026, 4, 1),
        min_degree=2,
        metadata_timestamp_path="review.timestamp",
        now=_dt(2026, 5, 11),
    )

    assert [row["unit"]["id"] for row in result] == ["hub-a", "hub-b"]
    assert result[0]["stale_source"] == "metadata"
    assert result[0]["metadata"]["timestamp_path"] == "review.timestamp"


def test_stale_hubs_respects_limit(store: Store):
    for unit in [
        _unit("older", "Older", updated_at=_dt(2025, 1, 1)),
        _unit("old", "Old", updated_at=_dt(2026, 1, 1)),
        _unit("leaf-a", "Leaf A", updated_at=_dt(2026, 5, 1)),
        _unit("leaf-b", "Leaf B", updated_at=_dt(2026, 5, 1)),
    ]:
        store.insert_unit(unit)
    for hub in ("older", "old"):
        _edge(store, hub, "leaf-a")
        _edge(store, "leaf-b", hub)

    result = GraphService(store).analyze_stale_hubs(
        _dt(2026, 4, 1),
        min_degree=2,
        limit=1,
        now=_dt(2026, 5, 11),
    )

    assert [row["unit"]["id"] for row in result] == ["older"]
