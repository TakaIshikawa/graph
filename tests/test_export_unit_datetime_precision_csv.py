from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from io import StringIO

from graph.export import export_unit_datetime_precision_csv
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, created_at: object = None, updated_at: object = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Source A",
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=[],
        created_at=created_at,
        updated_at=updated_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_datetime_precision_exports_core_and_metadata_temporal_values():
    text = export_unit_datetime_precision_csv(
        [
            unit(
                "b",
                created_at=datetime(2026, 4, 1, 12, 30, tzinfo=timezone.utc),
                updated_at=date(2026, 4, 2),
                metadata={
                    "published_date": "2026-04",
                    "starts_at": "2026-04-03T09:00:00-0700",
                    "description": "not temporal",
                },
            )
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "b",
            "title": "Title b",
            "source_project": "Source A",
            "source_entity_type": "note",
            "field_name": "created_at",
            "raw_value": "2026-04-01T12:30:00+00:00",
            "precision": "datetime",
            "timezone_present": "true",
        },
        {
            "unit_id": "b",
            "title": "Title b",
            "source_project": "Source A",
            "source_entity_type": "note",
            "field_name": "metadata.published_date",
            "raw_value": "2026-04",
            "precision": "year_month",
            "timezone_present": "false",
        },
        {
            "unit_id": "b",
            "title": "Title b",
            "source_project": "Source A",
            "source_entity_type": "note",
            "field_name": "metadata.starts_at",
            "raw_value": "2026-04-03T09:00:00-0700",
            "precision": "datetime",
            "timezone_present": "true",
        },
        {
            "unit_id": "b",
            "title": "Title b",
            "source_project": "Source A",
            "source_entity_type": "note",
            "field_name": "updated_at",
            "raw_value": "2026-04-02",
            "precision": "date",
            "timezone_present": "false",
        },
    ]


def test_unit_datetime_precision_classifies_common_string_precisions_and_malformed_values():
    text = export_unit_datetime_precision_csv(
        [
            unit(
                "a",
                metadata={
                    "date_values": ["2026", "2026-05", "2026-05-19", "2026-05-19T10:11Z", "May 2026"],
                    "timestamp": "",
                },
            )
        ]
    )

    assert [(row["raw_value"], row["precision"]) for row in rows(text)] == [
        ("2026", "year"),
        ("2026-05", "year_month"),
        ("2026-05-19", "date"),
        ("2026-05-19T10:11Z", "datetime"),
        ("May 2026", "unknown"),
    ]


def test_unit_datetime_precision_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "unit-datetime-precision.csv"
    units = [unit("a", metadata={"published": "2026"})]

    expected = export_unit_datetime_precision_csv(units)
    stats = export_unit_datetime_precision_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_unit_datetime_precision_is_deterministic_for_reversed_input():
    units = [
        unit("b", metadata={"published": "2026-05"}),
        unit("a", metadata={"published": "2026"}),
    ]

    assert export_unit_datetime_precision_csv(units) == export_unit_datetime_precision_csv(reversed(units))
