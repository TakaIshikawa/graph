from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(unit_id: str, title: str, created_at: datetime) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=title,
        content_type=ContentType.INSIGHT,
        created_at=created_at,
        ingested_at=created_at,
        updated_at=created_at,
    )


def edge(edge_id: str, from_id: str, to_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_id,
        to_unit_id=to_id,
        relation=EdgeRelation.RELATES_TO,
        source=EdgeSource.MANUAL,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_analyze_temporal_bridges_returns_sorted_qualifying_edges(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        old = store.insert_unit(unit("old", "Old", datetime(2020, 1, 1, tzinfo=timezone.utc)))
        mid = store.insert_unit(unit("mid", "Middle", datetime(2025, 1, 1, tzinfo=timezone.utc)))
        new = store.insert_unit(unit("new", "New", datetime(2026, 1, 1, tzinfo=timezone.utc)))
        store.insert_edge(edge("b-edge", old.id, new.id))
        store.insert_edge(edge("a-edge", old.id, mid.id))
        store.insert_edge(edge("short", mid.id, new.id))

        records = GraphService(store).analyze_temporal_bridges(min_gap_days=700, limit=2)

        assert [record["edge_id"] for record in records] == ["b-edge", "a-edge"]
        assert records[0]["from_unit"]["title"] == "Old"
        assert records[0]["to_unit"]["title"] == "New"
        assert records[0]["relation"] == "relates_to"
        assert records[0]["source"] == "manual"
        assert records[0]["gap_days"] > records[1]["gap_days"]
    finally:
        store.close()


def test_analyze_temporal_bridges_skips_missing_endpoint_units(tmp_path):
    old = unit("old", "Old", datetime(2020, 1, 1, tzinfo=timezone.utc))
    dangling = edge("dangling", old.id, "missing")

    class FakeStore:
        def get_all_edges(self):
            return [dangling]

        def get_unit(self, unit_id):
            return old if unit_id == old.id else None

    assert GraphService(FakeStore()).analyze_temporal_bridges(min_gap_days=1) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_gap_days": -1},
        {"limit": -1},
    ],
)
def test_analyze_temporal_bridges_validates_arguments(tmp_path, kwargs):
    store = Store(str(tmp_path / "graph.db"))
    try:
        with pytest.raises(ValueError):
            GraphService(store).analyze_temporal_bridges(**kwargs)
    finally:
        store.close()
