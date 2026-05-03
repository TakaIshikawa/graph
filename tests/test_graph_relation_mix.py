from __future__ import annotations

import os
import tempfile

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


def _unit(
    unit_id: str,
    source_project: SourceProject,
    title: str,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
    *,
    weight: float,
    source: EdgeSource,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=weight,
        source=source,
    )


def _by_key(records: list[dict], key: str, value: str) -> dict:
    return next(record for record in records if record[key] == value)


def test_analyze_relation_mix_reports_relation_and_edge_source_distribution(
    store: Store,
):
    for unit in [
        _unit("unit-a", SourceProject.MAX, "A"),
        _unit("unit-b", SourceProject.MAX, "B"),
        _unit("unit-c", SourceProject.FORTY_TWO, "C"),
        _unit("unit-d", SourceProject.PRESENCE, "D"),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge(
            "edge-a-c",
            "unit-a",
            "unit-c",
            EdgeRelation.REFERENCES,
            weight=2.0,
            source=EdgeSource.SOURCE,
        ),
        _edge(
            "edge-b-c",
            "unit-b",
            "unit-c",
            EdgeRelation.REFERENCES,
            weight=1.0,
            source=EdgeSource.INFERRED,
        ),
        _edge(
            "edge-c-a",
            "unit-c",
            "unit-a",
            EdgeRelation.BUILDS_ON,
            weight=3.0,
            source=EdgeSource.MANUAL,
        ),
        _edge(
            "edge-c-d",
            "unit-c",
            "unit-d",
            EdgeRelation.CONTAINS,
            weight=4.0,
            source=EdgeSource.SOURCE,
        ),
        _edge(
            "edge-d-a",
            "unit-d",
            "unit-a",
            EdgeRelation.CONTAINS,
            weight=2.0,
            source=EdgeSource.MANUAL,
        ),
    ]:
        store.insert_edge(edge)
    store.conn.execute("PRAGMA foreign_keys = OFF")
    store.conn.execute(
        """INSERT INTO edges
           (id, from_unit_id, to_unit_id, relation, weight, source, metadata, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "edge-missing-a",
            "missing-unit",
            "unit-a",
            EdgeRelation.CHALLENGES.value,
            5.0,
            EdgeSource.INFERRED.value,
            "{}",
            "2026-05-03T00:00:00+00:00",
        ),
    )
    store.conn.commit()
    store.conn.execute("PRAGMA foreign_keys = ON")

    result = GraphService(store).analyze_relation_mix()

    assert result["total_edge_count"] == 6
    assert result["total_weight"] == 17.0
    assert result["pair_edge_count"] == 5
    assert result["missing_endpoint_edge_count"] == 1

    references = _by_key(result["relation_mix"], "relation", "references")
    assert references == {
        "relation": "references",
        "edge_count": 2,
        "total_weight": 3.0,
        "edge_percentage": pytest.approx(33.333333),
        "weight_percentage": pytest.approx(17.647059),
    }
    contains = _by_key(result["relation_mix"], "relation", "contains")
    assert contains["edge_count"] == 2
    assert contains["total_weight"] == 6.0

    inferred = _by_key(result["edge_source_mix"], "source", "inferred")
    assert inferred["edge_count"] == 2
    assert inferred["total_weight"] == 6.0
    manual = _by_key(result["edge_source_mix"], "source", "manual")
    assert manual["edge_percentage"] == pytest.approx(33.333333)

    assert result["source_project_pairs"][0] == {
        "from_source_project": "max",
        "to_source_project": "forty_two",
        "edge_count": 2,
        "total_weight": 3.0,
        "edge_percentage": pytest.approx(33.333333),
        "weight_percentage": pytest.approx(17.647059),
        "relation_counts": {"references": 2},
        "edge_source_counts": {"inferred": 1, "source": 1},
    }


def test_analyze_relation_mix_top_pair_limit_and_order_are_deterministic(
    store: Store,
):
    for unit in [
        _unit("unit-a", SourceProject.MAX, "A"),
        _unit("unit-b", SourceProject.FORTY_TWO, "B"),
        _unit("unit-c", SourceProject.PRESENCE, "C"),
        _unit("unit-d", SourceProject.ME, "D"),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge(
            "edge-a-b",
            "unit-a",
            "unit-b",
            EdgeRelation.RELATES_TO,
            weight=1.0,
            source=EdgeSource.INFERRED,
        ),
        _edge(
            "edge-a-c",
            "unit-a",
            "unit-c",
            EdgeRelation.RELATES_TO,
            weight=1.0,
            source=EdgeSource.INFERRED,
        ),
        _edge(
            "edge-d-a",
            "unit-d",
            "unit-a",
            EdgeRelation.REFERENCES,
            weight=2.0,
            source=EdgeSource.SOURCE,
        ),
    ]:
        store.insert_edge(edge)

    result = GraphService(store).analyze_relation_mix(top_pair_limit=2)

    assert [
        (pair["from_source_project"], pair["to_source_project"])
        for pair in result["source_project_pairs"]
    ] == [("me", "max"), ("max", "forty_two")]


def test_analyze_relation_mix_can_skip_pair_payload(store: Store):
    store.insert_unit(_unit("unit-a", SourceProject.MAX, "A"))
    store.insert_unit(_unit("unit-b", SourceProject.FORTY_TWO, "B"))
    store.insert_edge(
        _edge(
            "edge-a-b",
            "unit-a",
            "unit-b",
            EdgeRelation.REFERENCES,
            weight=1.0,
            source=EdgeSource.SOURCE,
        )
    )

    result = GraphService(store).analyze_relation_mix(include_pairs=False)

    assert result["total_edge_count"] == 1
    assert result["source_project_pairs"] == []
    assert result["pair_edge_count"] == 0


def test_analyze_relation_mix_handles_empty_graph(store: Store):
    assert GraphService(store).analyze_relation_mix() == {
        "total_edge_count": 0,
        "total_weight": 0.0,
        "relation_mix": [],
        "edge_source_mix": [],
        "pair_edge_count": 0,
        "missing_endpoint_edge_count": 0,
        "source_project_pairs": [],
    }


@pytest.mark.parametrize("top_pair_limit", [-1, 1.5, "2", True])
def test_analyze_relation_mix_validates_top_pair_limit(
    store: Store,
    top_pair_limit: object,
):
    with pytest.raises(
        ValueError, match="top_pair_limit must be a non-negative integer"
    ):
        GraphService(store).analyze_relation_mix(top_pair_limit=top_pair_limit)  # type: ignore[arg-type]
