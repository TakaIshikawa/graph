from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=title,
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    weight: float,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=weight,
        source=EdgeSource.MANUAL,
        metadata={"weight_reason": edge_id},
    )


def test_edge_weight_outliers_returns_low_and_high_outliers_ordered_by_distance(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        for unit_id, title in (("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")):
            store.insert_unit(_unit(unit_id, title))
        store.insert_edge(_edge("low", "a", "b", 0.1))
        store.insert_edge(_edge("ok", "b", "c", 0.5))
        store.insert_edge(_edge("high", "a", "c", 1.8))

        rows = store.edge_weight_outliers(min_weight=0.25, max_weight=1.0)

        assert [row["id"] for row in rows] == ["high", "low"]
        assert rows[0]["from_title"] == "Alpha"
        assert rows[0]["to_title"] == "Gamma"
        assert rows[0]["metadata"] == {"weight_reason": "high"}
    finally:
        store.close()


def test_edge_weight_outliers_supports_relation_filter_limit_and_missing_titles(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        store.insert_unit(_unit("a", "Alpha"))
        store.insert_unit(_unit("b", "Beta"))
        store.insert_edge(_edge("relates", "a", "b", 2.0, EdgeRelation.RELATES_TO))
        store.conn.execute("PRAGMA foreign_keys = OFF")
        store.conn.execute(
            """INSERT INTO edges
               (id, from_unit_id, to_unit_id, relation, weight, source, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "missing",
                "a",
                "missing-unit",
                EdgeRelation.BUILDS_ON,
                3.0,
                EdgeSource.MANUAL,
                "{}",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        store.conn.commit()
        store.conn.execute("PRAGMA foreign_keys = ON")

        rows = store.edge_weight_outliers(
            max_weight=1.0,
            relation=EdgeRelation.BUILDS_ON,
            limit=1,
        )

        assert rows[0]["id"] == "missing"
        assert rows[0]["to_title"] is None
    finally:
        store.close()


def test_edge_weight_outliers_validates_arguments(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    try:
        with pytest.raises(ValueError, match="At least one"):
            store.edge_weight_outliers()
        with pytest.raises(ValueError, match="limit"):
            store.edge_weight_outliers(min_weight=0.1, limit=0)
        with pytest.raises(ValueError, match="min_weight"):
            store.edge_weight_outliers(min_weight=2.0, max_weight=1.0)
    finally:
        store.close()
