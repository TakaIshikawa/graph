from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.graph.service import GraphService, temporal_reachability
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


def _unit(unit_id: str, created_at: datetime | str | None) -> KnowledgeUnit:
    unit = KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="note",
        title=unit_id.upper(),
        content=f"{unit_id} content",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=[],
        confidence=None,
        utility_score=None,
        embedding=None,
        created_at=created_at,
        ingested_at=created_at,
        updated_at=created_at,
    )
    return unit


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    created_at: datetime | str | None,
    relation: EdgeRelation = EdgeRelation.REFERENCES,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=1.0,
        source=EdgeSource.MANUAL,
        metadata={},
        created_at=created_at,
    )


def test_temporal_reachability_returns_monotonic_paths_with_diagnostics():
    units = [
        _unit("a", _dt(2026, 1, 1)),
        _unit("b", _dt(2026, 1, 3)),
        _unit("c", _dt(2026, 1, 5)),
        _unit("older", _dt(2025, 12, 31)),
        _unit("missing", None),
        _unit("invalid", "not-a-date"),
    ]
    edges = [
        _edge("a-b", "a", "b", _dt(2026, 1, 2), EdgeRelation.BUILDS_ON),
        _edge("b-c", "b", "c", _dt(2026, 1, 4), EdgeRelation.REFERENCES),
        _edge("a-older", "a", "older", _dt(2026, 1, 2)),
        _edge("a-missing", "a", "missing", _dt(2026, 1, 2)),
        _edge("a-invalid", "a", "invalid", _dt(2026, 1, 2)),
        _edge("a-b-missing", "a", "b", None),
        _edge("a-b-invalid", "a", "b", "not-a-date"),
    ]

    result = temporal_reachability(
        units,
        edges,
        start_unit_id="a",
        end_unit_id="c",
        max_depth=2,
    )

    assert result["paths"] == [
        {
            "unit_ids": ["a", "b", "c"],
            "edge_ids": ["a-b", "b-c"],
            "relations": ["builds_on", "references"],
            "timestamps": [
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
                "2026-01-03T00:00:00+00:00",
                "2026-01-04T00:00:00+00:00",
                "2026-01-05T00:00:00+00:00",
            ],
            "depth": 2,
        }
    ]
    assert result["diagnostics"]["missing_unit_timestamps"] == 1
    assert result["diagnostics"]["invalid_unit_timestamps"] == 1
    assert result["diagnostics"]["missing_edge_timestamps"] == 1
    assert result["diagnostics"]["invalid_edge_timestamps"] == 1
    assert result["diagnostics"]["skipped_non_monotonic"] == 1


def test_temporal_reachability_can_disable_monotonic_ordering():
    units = [_unit("a", _dt(2026, 1, 2)), _unit("b", _dt(2026, 1, 1))]
    edges = [_edge("a-b", "a", "b", _dt(2026, 1, 3))]

    monotonic = temporal_reachability(units, edges, start_unit_id="a")
    non_monotonic = temporal_reachability(
        units,
        edges,
        start_unit_id="a",
        monotonic=False,
    )

    assert monotonic["paths"] == []
    assert non_monotonic["paths"][0]["unit_ids"] == ["a", "b"]


def test_temporal_reachability_validates_inputs():
    with pytest.raises(ValueError, match="Unsupported timestamp field"):
        temporal_reachability([], [], start_unit_id="a", timestamp_field="bogus")
    with pytest.raises(ValueError, match="direction must be"):
        temporal_reachability([], [], start_unit_id="a", direction="sideways")
    with pytest.raises(ValueError, match="max_depth must be"):
        temporal_reachability([], [], start_unit_id="a", max_depth=-1)


def test_graph_service_temporal_reachability_uses_store(store: Store):
    for unit in [_unit("a", _dt(2026, 1, 1)), _unit("b", _dt(2026, 1, 3))]:
        store.insert_unit(unit)
    store.insert_edge(_edge("a-b", "a", "b", _dt(2026, 1, 2)))

    result = GraphService(store).temporal_reachability(start_unit_id="a")

    assert result["path_count"] == 1
    assert result["paths"][0]["unit_ids"] == ["a", "b"]
