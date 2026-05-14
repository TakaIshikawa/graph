from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_relation_date_lag_csv
from graph.types.enums import EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, created_at: object = UNIT_TIME, metadata: dict | None = None) -> KnowledgeUnit:
    item = KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        tags=[],
        created_at=created_at,
        ingested_at=None,
        updated_at=None,
    )
    return item


def edge(edge_id: str, source: str, target: str, relation: object = EdgeRelation.RELATES_TO) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=source,
        to_unit_id=target,
        relation=relation,
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_relation_date_lag_csv_empty_edges_returns_header():
    assert export_relation_date_lag_csv([], []) == (
        "relation,source_unit_id,target_unit_id,source_date,target_date,lag_days,lag_bucket\n"
    )


def test_export_relation_date_lag_csv_sorts_by_relation_source_and_target():
    text = export_relation_date_lag_csv(
        [unit("a"), unit("b")],
        [edge("2", "b", "a", "zeta"), edge("1", "a", "b", "alpha")],
    )

    assert [(row["relation"], row["source_unit_id"]) for row in rows(text)] == [
        ("alpha", "a"),
        ("zeta", "b"),
    ]


def test_export_relation_date_lag_csv_computes_signed_lag_days():
    text = export_relation_date_lag_csv(
        [
            unit("a", created_at="2026-01-10"),
            unit("b", created_at="2026-01-12"),
            unit("c", created_at="2026-01-09"),
        ],
        [edge("ab", "a", "b"), edge("ac", "a", "c")],
    )

    assert [(row["lag_days"], row["lag_bucket"]) for row in rows(text)] == [
        ("2", "source_before_target"),
        ("-1", "source_after_target"),
    ]


def test_export_relation_date_lag_csv_uses_metadata_dates_and_unknown_bucket():
    text = export_relation_date_lag_csv(
        [unit("a", metadata={"published_at": "2026-01-01T12:00:00Z"})],
        [edge("ab", "a", "missing")],
    )

    assert rows(text)[0]["source_date"] == "2026-01-01"
    assert rows(text)[0]["target_date"] == ""
    assert rows(text)[0]["lag_bucket"] == "unknown"


def test_export_relation_date_lag_csv_path_mode(tmp_path):
    units = [unit("a", created_at="2026-01-01"), unit("b", created_at="2026-01-01")]
    edges = [edge("ab", "a", "b")]
    path = tmp_path / "lags.csv"

    stats = export_relation_date_lag_csv(units, edges, path)

    assert rows(path.read_text(encoding="utf-8"))[0]["lag_bucket"] == "same_day"
    assert stats["unit_count"] == 2
    assert stats["edge_count"] == 1
    assert stats["rows_exported"] == 1
