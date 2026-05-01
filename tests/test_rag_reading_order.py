from __future__ import annotations

import pytest

from graph.rag import plan_reading_order
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(unit_id: str, title: str | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title or unit_id,
        content=f"Content for {unit_id}",
        content_type=ContentType.INSIGHT,
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


def test_plan_reading_order_orders_prerequisites_before_dependents_in_dag():
    units = [
        unit("unit-implementation", "Implementation"),
        unit("unit-foundation", "Foundation"),
        unit("unit-paper", "Paper"),
        unit("unit-course", "Course"),
    ]
    edges = [
        edge(
            "edge-implementation-foundation",
            "unit-implementation",
            "unit-foundation",
            EdgeRelation.BUILDS_ON,
        ),
        edge(
            "edge-implementation-paper",
            "unit-implementation",
            "unit-paper",
            EdgeRelation.REFERENCES,
        ),
        edge("edge-course-paper", "unit-course", "unit-paper", EdgeRelation.CONTAINS),
    ]

    result = plan_reading_order(reversed(units), reversed(edges))

    assert unit_ids(result) == [
        "unit-course",
        "unit-foundation",
        "unit-paper",
        "unit-implementation",
    ]
    assert result["units"][0] == {
        "id": "unit-course",
        "source_project": "max",
        "source_id": "source-unit-course",
        "source_entity_type": "insight",
        "title": "Course",
        "content_type": "insight",
        "reason": "prerequisite",
    }
    assert result["stats"] == {
        "total_units": 4,
        "planned_units": 4,
        "candidate_units": 4,
        "omitted_units": 0,
        "seed_unit_id": None,
        "seed_found": False,
        "cycles_detected": False,
        "cycle_fallback_count": 0,
        "cycle_fallback_unit_ids": [],
        "limit": None,
    }


def test_plan_reading_order_reports_cycles_and_breaks_ties_deterministically():
    units = [unit("unit-c"), unit("unit-a"), unit("unit-b"), unit("unit-d")]
    edges = [
        edge("edge-a-b", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
        edge("edge-b-c", "unit-b", "unit-c", EdgeRelation.BUILDS_ON),
        edge("edge-c-a", "unit-c", "unit-a", EdgeRelation.BUILDS_ON),
        edge("edge-d-c", "unit-d", "unit-c", EdgeRelation.DERIVES_FROM),
    ]

    first = plan_reading_order(units, edges)
    second = plan_reading_order(reversed(units), reversed(edges))

    assert first == second
    assert unit_ids(first) == ["unit-a", "unit-c", "unit-b", "unit-d"]
    assert first["units"][0]["reason"] == "cycle_fallback"
    assert first["stats"]["cycles_detected"] is True
    assert first["stats"]["cycle_fallback_count"] == 1
    assert first["stats"]["cycle_fallback_unit_ids"] == ["unit-a"]


def test_plan_reading_order_handles_disconnected_components_without_seed():
    units = [unit("unit-a"), unit("unit-b"), unit("unit-x"), unit("unit-y")]
    edges = [
        edge("edge-b-a", "unit-b", "unit-a", EdgeRelation.BUILDS_ON),
        edge("edge-y-x", "unit-y", "unit-x", EdgeRelation.REFERENCES),
    ]

    result = plan_reading_order(units, edges)

    assert unit_ids(result) == ["unit-a", "unit-b", "unit-x", "unit-y"]
    assert result["stats"]["candidate_units"] == 4
    assert result["stats"]["omitted_units"] == 0


def test_plan_reading_order_seed_filters_to_connected_component_and_marks_reasons():
    units = [unit("unit-a"), unit("unit-b"), unit("unit-c"), unit("unit-x"), unit("unit-y")]
    edges = [
        edge("edge-b-a", "unit-b", "unit-a", EdgeRelation.BUILDS_ON),
        edge("edge-c-b", "unit-c", "unit-b", EdgeRelation.REFERENCES),
        edge("edge-y-x", "unit-y", "unit-x", EdgeRelation.BUILDS_ON),
    ]

    result = plan_reading_order(units, edges, seed_unit_id="unit-b")

    assert unit_ids(result) == ["unit-a", "unit-b", "unit-c"]
    assert [item["reason"] for item in result["units"]] == ["neighbor", "seed", "neighbor"]
    assert result["stats"]["total_units"] == 5
    assert result["stats"]["candidate_units"] == 3
    assert result["stats"]["seed_unit_id"] == "unit-b"
    assert result["stats"]["seed_found"] is True


def test_plan_reading_order_returns_empty_when_seed_is_missing():
    result = plan_reading_order([unit("unit-a")], [], seed_unit_id="unit-missing")

    assert result["units"] == []
    assert result["stats"]["candidate_units"] == 0
    assert result["stats"]["seed_unit_id"] == "unit-missing"
    assert result["stats"]["seed_found"] is False


def test_plan_reading_order_limit_is_deterministic_and_reported_in_stats():
    units = [unit("unit-c"), unit("unit-a"), unit("unit-b")]
    edges = [
        edge("edge-b-a", "unit-b", "unit-a", EdgeRelation.BUILDS_ON),
        edge("edge-c-b", "unit-c", "unit-b", EdgeRelation.BUILDS_ON),
    ]

    first = plan_reading_order(units, edges, limit=2)
    second = plan_reading_order(reversed(units), reversed(edges), limit=2)

    assert first == second
    assert unit_ids(first) == ["unit-a", "unit-b"]
    assert first["stats"]["planned_units"] == 2
    assert first["stats"]["candidate_units"] == 3
    assert first["stats"]["omitted_units"] == 1
    assert first["stats"]["limit"] == 2


def test_plan_reading_order_accepts_zero_limit():
    result = plan_reading_order([unit("unit-a")], [], limit=0)

    assert result["units"] == []
    assert result["stats"]["planned_units"] == 0
    assert result["stats"]["candidate_units"] == 1
    assert result["stats"]["omitted_units"] == 1


@pytest.mark.parametrize("limit", [-1, "2", None])
def test_plan_reading_order_validates_limit(limit):
    kwargs = {} if limit is None else {"limit": limit}
    if limit is None:
        assert plan_reading_order([], [], **kwargs)["units"] == []
    else:
        with pytest.raises(ValueError, match="limit must be a non-negative integer"):
            plan_reading_order([], [], **kwargs)
