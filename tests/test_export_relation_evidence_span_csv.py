from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from io import StringIO
from types import SimpleNamespace

from graph.export import export_relation_evidence_span_csv
from graph.types.enums import EdgeRelation
from graph.types.models import KnowledgeEdge


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_relation_evidence_span_csv_empty_input_returns_header():
    assert export_relation_evidence_span_csv([]) == (
        "relation_type,source_id,target_id,evidence_count,first_evidence_date,last_evidence_date,"
        "evidence_span_days,has_multi_date_span\n"
    )


def test_export_relation_evidence_span_csv_handles_edge_metadata_and_nested_evidence_dates():
    edge = KnowledgeEdge.model_construct(
        from_unit_id="a",
        to_unit_id="b",
        relation=EdgeRelation.RELATES_TO,
        metadata={
            "observed_at": "2026-01-02T10:00:00Z",
            "evidence": [
                {"date": date(2026, 1, 1)},
                {"metadata": {"evidence_date": datetime(2026, 1, 5, tzinfo=timezone.utc)}},
                {"date": "not-a-date"},
            ],
        },
    )

    row = rows(export_relation_evidence_span_csv([edge]))[0]
    assert row["relation_type"] == "relates_to"
    assert row["source_id"] == "a"
    assert row["target_id"] == "b"
    assert row["evidence_count"] == "3"
    assert row["first_evidence_date"] == "2026-01-01"
    assert row["last_evidence_date"] == "2026-01-05"
    assert row["evidence_span_days"] == "4"
    assert row["has_multi_date_span"] == "true"


def test_export_relation_evidence_span_csv_accepts_mapping_and_relation_like_objects():
    mapping_edge = {
        "relation_type": "supports",
        "source_id": "s",
        "target_id": "t",
        "attributes": {"evidence_items": [{"observed_date": "2026-02-01"}]},
    }
    object_edge = SimpleNamespace(
        relation="mentions",
        from_unit_id="a",
        to_unit_id="c",
        evidence_dates=["2026-03-01T00:00:00Z", "2026-03-01"],
    )

    result = rows(export_relation_evidence_span_csv([mapping_edge, object_edge]))
    assert [(row["relation_type"], row["evidence_count"]) for row in result] == [
        ("mentions", "2"),
        ("supports", "1"),
    ]
    assert result[0]["has_multi_date_span"] == "false"


def test_export_relation_evidence_span_csv_includes_undated_relations():
    row = rows(
        export_relation_evidence_span_csv(
            [{"relation": "blocks", "from_unit_id": "a", "to_unit_id": "b", "metadata": {"date": "bad"}}]
        )
    )[0]

    assert row["evidence_count"] == "0"
    assert row["first_evidence_date"] == ""
    assert row["last_evidence_date"] == ""
    assert row["evidence_span_days"] == ""
    assert row["has_multi_date_span"] == "false"


def test_export_relation_evidence_span_csv_sorts_by_relation_source_and_target():
    text = export_relation_evidence_span_csv(
        [
            {"relation": "zeta", "from_unit_id": "b", "to_unit_id": "a"},
            {"relation": "alpha", "from_unit_id": "c", "to_unit_id": "a"},
            {"relation": "alpha", "from_unit_id": "a", "to_unit_id": "z"},
        ]
    )

    assert [(row["relation_type"], row["source_id"]) for row in rows(text)] == [
        ("alpha", "a"),
        ("alpha", "c"),
        ("zeta", "b"),
    ]


def test_export_relation_evidence_span_csv_path_mode(tmp_path):
    path = tmp_path / "spans.csv"
    stats = export_relation_evidence_span_csv(
        [
            {"relation": "a", "source_id": "s", "target_id": "t", "date": "2026-01-01"},
            {"relation": "b", "source_id": "s", "target_id": "u"},
        ],
        path,
    )

    assert rows(path.read_text(encoding="utf-8"))[0]["first_evidence_date"] == "2026-01-01"
    assert stats["path"] == str(path)
    assert stats["relation_count"] == 2
    assert stats["dated_relation_count"] == 1
    assert stats["rows_exported"] == 2
    assert stats["bytes_written"] == path.stat().st_size
