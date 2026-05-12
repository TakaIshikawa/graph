from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.store.db import Store
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def _dt(day: int, hour: int = 0) -> datetime:
    return datetime(2024, 1, day, hour, tzinfo=timezone.utc)


def _unit(unit_id: str, created_at: datetime) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        created_at=created_at,
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
    *,
    source: EdgeSource = EdgeSource.INFERRED,
    created_at: datetime,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=source,
        created_at=created_at,
    )


def test_edge_relation_age_profile_empty_store_returns_empty(store: Store):
    assert store.edge_relation_age_profile() == []


def test_edge_relation_age_profile_groups_and_orders_deterministically(store: Store):
    store.insert_unit(_unit("a", _dt(1)))
    store.insert_unit(_unit("b", _dt(3)))
    store.insert_unit(_unit("c", _dt(4)))
    store.insert_unit(_unit("d", _dt(5)))
    store.insert_edge(_edge("build-2", "b", "c", EdgeRelation.BUILDS_ON, created_at=_dt(7)))
    store.insert_edge(_edge("ref-1", "a", "b", EdgeRelation.REFERENCES, created_at=_dt(5, 12)))
    store.insert_edge(_edge("build-1", "a", "b", EdgeRelation.BUILDS_ON, created_at=_dt(6)))
    store.insert_edge(_edge("rel-1", "c", "d", EdgeRelation.RELATES_TO, created_at=_dt(8)))

    assert store.edge_relation_age_profile() == [
        {
            "relation": "builds_on",
            "edge_count": 2,
            "average_age_days": 3.0,
            "min_age_days": 3.0,
            "max_age_days": 3.0,
            "example_edge_ids": ["build-1", "build-2"],
        },
        {
            "relation": "references",
            "edge_count": 1,
            "average_age_days": 2.5,
            "min_age_days": 2.5,
            "max_age_days": 2.5,
            "example_edge_ids": ["ref-1"],
        },
        {
            "relation": "relates_to",
            "edge_count": 1,
            "average_age_days": 3.0,
            "min_age_days": 3.0,
            "max_age_days": 3.0,
            "example_edge_ids": ["rel-1"],
        },
    ]


def test_edge_relation_age_profile_filters_and_ignores_dangling_edges(store: Store):
    store.insert_unit(_unit("a", _dt(1)))
    store.insert_unit(_unit("b", _dt(2)))
    store.insert_unit(_unit("c", _dt(3)))
    store.insert_edge(
        _edge(
            "manual-build",
            "a",
            "b",
            EdgeRelation.BUILDS_ON,
            source=EdgeSource.MANUAL,
            created_at=_dt(5),
        )
    )
    store.insert_edge(_edge("inferred-build", "b", "c", EdgeRelation.BUILDS_ON, created_at=_dt(8)))
    store.insert_edge(_edge("manual-ref", "a", "c", EdgeRelation.REFERENCES, source=EdgeSource.MANUAL, created_at=_dt(4)))
    store.conn.execute("PRAGMA foreign_keys = OFF")
    store.conn.execute(
        """INSERT INTO edges
           (id, from_unit_id, to_unit_id, relation, weight, source, metadata, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("dangling", "a", "missing", "builds_on", 1.0, "inferred", "{}", _dt(9).isoformat()),
    )
    store.conn.commit()
    store.conn.execute("PRAGMA foreign_keys = ON")

    assert store.edge_relation_age_profile(relation=EdgeRelation.BUILDS_ON) == [
        {
            "relation": "builds_on",
            "edge_count": 2,
            "average_age_days": 4.0,
            "min_age_days": 3.0,
            "max_age_days": 5.0,
            "example_edge_ids": ["manual-build", "inferred-build"],
        }
    ]
    assert store.edge_relation_age_profile(source=EdgeSource.MANUAL, min_edge_count=2) == []
    assert store.edge_relation_age_profile(source=EdgeSource.MANUAL, limit=1) == [
        {
            "relation": "builds_on",
            "edge_count": 1,
            "average_age_days": 3.0,
            "min_age_days": 3.0,
            "max_age_days": 3.0,
            "example_edge_ids": ["manual-build"],
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_edge_count": 0}, "min_edge_count must be a positive integer"),
        ({"min_edge_count": True}, "min_edge_count must be a positive integer"),
        ({"limit": 0}, "limit must be a positive integer or None"),
        ({"limit": True}, "limit must be a positive integer or None"),
    ],
)
def test_edge_relation_age_profile_validates_options(store: Store, kwargs, message):
    with pytest.raises(ValueError, match=message):
        store.edge_relation_age_profile(**kwargs)
