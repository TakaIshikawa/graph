from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_edge_temporal_lag_csv
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

_MISSING = object()


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    *,
    relation: EdgeRelation | str = EdgeRelation.RELATES_TO,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.INFERRED,
        weight=1.0,
        metadata={},
    )


def unit(unit_id: str, *, metadata: dict | None = None, created_at: object = _MISSING) -> KnowledgeUnit:
    payload = {
        "id": unit_id,
        "source_project": SourceProject.CSV,
        "source_id": f"source-{unit_id}",
        "source_entity_type": "item",
        "title": unit_id,
        "content": "",
        "content_type": ContentType.INSIGHT,
        "metadata": metadata or {},
    }
    if created_at is not _MISSING:
        payload["created_at"] = created_at
        if created_at is None:
            payload["updated_at"] = None
            payload["ingested_at"] = None
    return KnowledgeUnit.model_construct(**payload)


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_temporal_lag_csv_empty_input_returns_header():
    assert export_edge_temporal_lag_csv([], []) == (
        "edge_id,relation,from_unit_id,to_unit_id,from_date,to_date,lag_days,lag_direction\n"
    )


def test_edge_temporal_lag_csv_emits_signed_lag_days_and_direction():
    text = export_edge_temporal_lag_csv(
        [
            edge("before", "u1", "u2", relation=EdgeRelation.REFERENCES),
            edge("after", "u2", "u1", relation=EdgeRelation.BUILDS_ON),
            edge("same", "u1", "u3", relation=EdgeRelation.RELATES_TO),
        ],
        [
            unit("u1", metadata={"date": "2024-01-01"}),
            unit("u2", metadata={"published_at": "2024-01-04T12:00:00Z"}),
            unit("u3", metadata={"source_date": "2024-01-01"}),
        ],
    )

    assert rows(text) == [
        {
            "edge_id": "after",
            "relation": "builds_on",
            "from_unit_id": "u2",
            "to_unit_id": "u1",
            "from_date": "2024-01-04",
            "to_date": "2024-01-01",
            "lag_days": "-3",
            "lag_direction": "after",
        },
        {
            "edge_id": "before",
            "relation": "references",
            "from_unit_id": "u1",
            "to_unit_id": "u2",
            "from_date": "2024-01-01",
            "to_date": "2024-01-04",
            "lag_days": "3",
            "lag_direction": "before",
        },
        {
            "edge_id": "same",
            "relation": "relates_to",
            "from_unit_id": "u1",
            "to_unit_id": "u3",
            "from_date": "2024-01-01",
            "to_date": "2024-01-01",
            "lag_days": "0",
            "lag_direction": "same",
        },
    ]


def test_edge_temporal_lag_csv_falls_back_to_unit_fields_and_includes_missing_dates():
    text = export_edge_temporal_lag_csv(
        [edge("a", "u1", "missing"), edge("b", "u1", "u2")],
        [
            unit("u1", created_at=datetime(2024, 2, 1, tzinfo=timezone.utc)),
            unit("u2", metadata={"date": "not-a-date"}, created_at=None),
        ],
    )

    assert rows(text) == [
        {
            "edge_id": "a",
            "relation": "relates_to",
            "from_unit_id": "u1",
            "to_unit_id": "missing",
            "from_date": "2024-02-01",
            "to_date": "",
            "lag_days": "",
            "lag_direction": "missing_date",
        },
        {
            "edge_id": "b",
            "relation": "relates_to",
            "from_unit_id": "u1",
            "to_unit_id": "u2",
            "from_date": "2024-02-01",
            "to_date": "",
            "lag_days": "",
            "lag_direction": "missing_date",
        },
    ]


def test_edge_temporal_lag_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "temporal-lag.csv"
    edges = [edge("a", "u1", "u2")]
    units = [unit("u1", metadata={"date": "2024-01-01"}), unit("u2", metadata={"date": "2024-01-02"})]

    expected = export_edge_temporal_lag_csv(edges, units)
    stats = export_edge_temporal_lag_csv(edges, units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "unit_count": 2,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
