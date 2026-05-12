from __future__ import annotations

import csv
from io import StringIO

from graph.export.relation_temporal_lag_csv import export_relation_temporal_lag_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    *,
    relation: EdgeRelation | str = EdgeRelation.RELATES_TO,
    created_at: object = None,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=EdgeSource.INFERRED,
        weight=1.0,
        metadata=metadata or {},
        created_at=created_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_temporal_lag_csv_empty_input_has_header_only():
    assert export_relation_temporal_lag_csv([]) == (
        "edge_id,from_unit_id,to_unit_id,relation,edge_created_date,evidence_date,"
        "lag_days,lag_bucket,source_project,source_entity_type\n"
    )


def test_relation_temporal_lag_csv_reports_lag_from_metadata_evidence_date():
    text = export_relation_temporal_lag_csv(
        [
            edge(
                "a",
                relation=EdgeRelation.REFERENCES,
                created_at="2024-02-10T09:00:00Z",
                metadata={
                    "source_project": "Project A",
                    "source_entity_type": "issue",
                    "observed_at": "not a date",
                    "observed_date": "2024-02-03T10:00:00Z",
                    "published_at": "2024-02-01",
                },
            )
        ]
    )

    assert rows(text) == [
        {
            "edge_id": "a",
            "from_unit_id": "from-a",
            "to_unit_id": "to-a",
            "relation": "references",
            "edge_created_date": "2024-02-10",
            "evidence_date": "2024-02-03",
            "lag_days": "7",
            "lag_bucket": "week",
            "source_project": "Project A",
            "source_entity_type": "issue",
        }
    ]


def test_relation_temporal_lag_csv_includes_edges_without_evidence_dates():
    text = export_relation_temporal_lag_csv(
        [
            edge(
                "undated",
                relation="custom",
                created_at="2024-02-10",
                metadata={"source_project": "Project B", "date": "unknown"},
            )
        ]
    )

    assert rows(text) == [
        {
            "edge_id": "undated",
            "from_unit_id": "from-undated",
            "to_unit_id": "to-undated",
            "relation": "custom",
            "edge_created_date": "2024-02-10",
            "evidence_date": "",
            "lag_days": "",
            "lag_bucket": "undated",
            "source_project": "Project B",
            "source_entity_type": "Unknown",
        }
    ]


def test_relation_temporal_lag_csv_buckets_same_day_month_and_future_evidence():
    text = export_relation_temporal_lag_csv(
        [
            edge("same", created_at="2024-01-01", metadata={"date": "2024-01-01"}),
            edge("month", created_at="2024-02-01", metadata={"date": "2024-01-10"}),
            edge("future", created_at="2024-01-01", metadata={"date": "2024-01-03"}),
        ]
    )

    by_id = {row["edge_id"]: row for row in rows(text)}
    assert by_id["same"]["lag_days"] == "0"
    assert by_id["same"]["lag_bucket"] == "same_day"
    assert by_id["month"]["lag_days"] == "22"
    assert by_id["month"]["lag_bucket"] == "month"
    assert by_id["future"]["lag_days"] == "-2"
    assert by_id["future"]["lag_bucket"] == "week"


def test_relation_temporal_lag_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "relation-temporal-lag.csv"
    edges = [edge("a", created_at="2024-01-02", metadata={"date": "2024-01-01"})]

    expected = export_relation_temporal_lag_csv(edges)
    stats = export_relation_temporal_lag_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
