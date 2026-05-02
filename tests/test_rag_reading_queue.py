from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.rag import build_reading_queue
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


NOW = datetime(2026, 5, 2, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str | None = None,
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
    updated_at: datetime | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title or unit_id,
        content=f"Content for {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
        updated_at=updated_at or NOW,
        created_at=updated_at or NOW,
    )


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.INFERRED,
    )


def unit_ids(result: dict) -> list[str]:
    return [item["id"] for item in result["units"]]


def test_build_reading_queue_prioritizes_unread_high_priority_units():
    units = [
        unit(
            "low-read",
            "Read Low",
            metadata={"priority": "low", "read_status": "read", "last_read_at": NOW},
        ),
        unit("high-unread", "High Unread", metadata={"priority": "high", "read_status": "unread"}),
        unit("medium-unread", "Medium Unread", metadata={"priority": "medium", "read_status": "unread"}),
    ]

    result = build_reading_queue(reversed(units), now=NOW)

    assert unit_ids(result) == ["high-unread", "medium-unread", "low-read"]
    assert result["units"][0]["score"] > result["units"][1]["score"]
    assert "high priority" in result["units"][0]["explanation"]
    assert "unread" in result["units"][0]["explanation"]
    assert result["stats"]["queued_units"] == 3


def test_build_reading_queue_boosts_stale_and_updated_since_last_read_units():
    old_read = datetime(2025, 1, 1, tzinfo=timezone.utc)
    recently_read = datetime(2026, 4, 20, tzinfo=timezone.utc)
    updated_after_read = datetime(2026, 5, 1, tzinfo=timezone.utc)
    units = [
        unit(
            "recent",
            metadata={"read_status": "read", "last_read_at": recently_read},
            updated_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        ),
        unit(
            "stale",
            metadata={"read_status": "read", "last_read_at": old_read},
            updated_at=old_read,
        ),
        unit(
            "changed",
            metadata={"read_status": "read", "last_read_at": recently_read},
            updated_at=updated_after_read,
        ),
    ]

    result = build_reading_queue(units, now=NOW)

    assert unit_ids(result)[:2] == ["changed", "stale"]
    assert "updated since last read" in result["units"][0]["explanation"]
    assert "stale" in result["units"][1]["explanation"]


def test_build_reading_queue_boosts_units_referenced_by_inbound_edges():
    units = [
        unit("source-a", metadata={"read_status": "read", "last_read_at": NOW}),
        unit("source-b", metadata={"read_status": "read", "last_read_at": NOW}),
        unit("target", metadata={"read_status": "read", "last_read_at": NOW}),
    ]
    edges = [
        edge("edge-a-target", "source-a", "target", EdgeRelation.REFERENCES),
        edge("edge-b-target", "source-b", "target", EdgeRelation.BUILDS_ON),
        edge("edge-target-a", "target", "source-a", EdgeRelation.RELATES_TO),
    ]

    result = build_reading_queue(units, edges, now=NOW)

    assert unit_ids(result)[0] == "target"
    assert result["units"][0]["inbound_reference_count"] == 2
    assert "referenced by 2 unit(s)" in result["units"][0]["explanation"]
    assert result["stats"]["edge_boosted_units"] == 1


def test_build_reading_queue_limit_zero_returns_no_units_with_stats():
    result = build_reading_queue(
        [
            unit("unit-a", metadata={"read_status": "unread"}),
            unit("unit-b", metadata={"read_status": "unread"}),
        ],
        limit=0,
        now=NOW,
    )

    assert result["units"] == []
    assert result["stats"]["total_units"] == 2
    assert result["stats"]["candidate_units"] == 2
    assert result["stats"]["queued_units"] == 0
    assert result["stats"]["omitted_units"] == 2
    assert result["stats"]["limit"] == 0


def test_build_reading_queue_orders_ties_by_title_then_id_deterministically():
    units = [
        unit("unit-c", "Same", metadata={"read_status": "unread"}),
        unit("unit-a", "Alpha", metadata={"read_status": "unread"}),
        unit("unit-b", "Same", metadata={"read_status": "unread"}),
    ]

    first = build_reading_queue(units, now=NOW)
    second = build_reading_queue(reversed(units), now=NOW)

    assert first == second
    assert unit_ids(first) == ["unit-a", "unit-b", "unit-c"]


def test_build_reading_queue_is_importable_from_graph_rag():
    from graph.rag import build_reading_queue as imported

    assert imported is build_reading_queue


@pytest.mark.parametrize("limit", [-1, "2", True])
def test_build_reading_queue_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        build_reading_queue([], limit=limit)
