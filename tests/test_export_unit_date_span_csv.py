from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from io import StringIO

from graph.export.unit_date_span_csv import export_unit_date_span_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    source_project: SourceProject | str | None,
    source_entity_type: str | None,
    *,
    created_at: object,
    ingested_at: object,
    updated_at: object,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={},
        created_at=created_at,
        ingested_at=ingested_at,
        updated_at=updated_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_date_span_exports_lifecycle_dates_and_spans():
    text = export_unit_date_span_csv(
        [
            unit(
                "b",
                SourceProject.MAX,
                "note",
                created_at=datetime(2026, 5, 1, 23, 30, tzinfo=timezone.utc),
                ingested_at=date(2026, 5, 3),
                updated_at="2026-05-10T04:30:00Z",
            ),
            unit(
                "a",
                None,
                None,
                created_at="2026-04-01",
                ingested_at="2026-04-01T12:00:00+00:00",
                updated_at="2026-04-02",
            ),
        ]
    )

    assert text.splitlines()[0] == (
        "unit_id,source_project,source_entity_type,created_date,ingested_date,"
        "updated_date,created_to_ingested_days,created_to_updated_days,ingested_to_updated_days"
    )
    assert rows(text) == [
        {
            "unit_id": "b",
            "source_project": "max",
            "source_entity_type": "note",
            "created_date": "2026-05-01",
            "ingested_date": "2026-05-03",
            "updated_date": "2026-05-10",
            "created_to_ingested_days": "2",
            "created_to_updated_days": "9",
            "ingested_to_updated_days": "7",
        },
        {
            "unit_id": "a",
            "source_project": "Unknown",
            "source_entity_type": "Unknown",
            "created_date": "2026-04-01",
            "ingested_date": "2026-04-01",
            "updated_date": "2026-04-02",
            "created_to_ingested_days": "0",
            "created_to_updated_days": "1",
            "ingested_to_updated_days": "1",
        },
    ]


def test_unit_date_span_leaves_dates_and_spans_blank_for_missing_or_invalid_timestamps():
    text = export_unit_date_span_csv(
        [
            unit(
                "a",
                "Source A",
                "note",
                created_at="not a date",
                ingested_at=None,
                updated_at="2026-05-10",
            ),
            unit(
                "b",
                "Source A",
                "note",
                created_at="2026-05-10",
                ingested_at="bad",
                updated_at="2026-05-09",
            ),
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "source_project": "Source A",
            "source_entity_type": "note",
            "created_date": "",
            "ingested_date": "",
            "updated_date": "2026-05-10",
            "created_to_ingested_days": "",
            "created_to_updated_days": "",
            "ingested_to_updated_days": "",
        },
        {
            "unit_id": "b",
            "source_project": "Source A",
            "source_entity_type": "note",
            "created_date": "2026-05-10",
            "ingested_date": "",
            "updated_date": "2026-05-09",
            "created_to_ingested_days": "",
            "created_to_updated_days": "-1",
            "ingested_to_updated_days": "",
        },
    ]


def test_unit_date_span_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "unit-date-span.csv"
    units = [
        unit(
            "a",
            "Source A",
            "note",
            created_at="2026-05-01",
            ingested_at="2026-05-02",
            updated_at="2026-05-03",
        )
    ]

    expected = export_unit_date_span_csv(units)
    stats = export_unit_date_span_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_unit_date_span_is_deterministic_for_reversed_input():
    units = [
        unit("b", "Source B", "note", created_at="2026-05-01", ingested_at="2026-05-02", updated_at="2026-05-03"),
        unit("a", "Source A", "note", created_at="2026-05-01", ingested_at="2026-05-02", updated_at="2026-05-03"),
    ]

    assert export_unit_date_span_csv(units) == export_unit_date_span_csv(reversed(units))
