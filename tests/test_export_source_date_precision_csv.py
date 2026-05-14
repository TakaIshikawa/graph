from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from io import StringIO

from graph.export import export_source_date_precision_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    source_project: SourceProject | str | None,
    *,
    created_at: object = None,
    updated_at: object = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
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


def test_source_date_precision_empty_input_returns_header_only_csv():
    assert export_source_date_precision_csv([]) == (
        "source_project,unit_count,dated_unit_count,undated_unit_count,date_only_count,"
        "datetime_count,year_month_count,year_only_count,date_coverage_percent\n"
    )


def test_source_date_precision_groups_by_source_and_unknown():
    text = export_source_date_precision_csv(
        [
            unit("a", SourceProject.MAX, created_at=date(2024, 1, 2)),
            unit("b", SourceProject.MAX, metadata={"published_date": "2024-05"}),
            unit("c", SourceProject.MAX, metadata={"year": "2024"}),
            unit("d", None, metadata={"date": "not a date"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "unit_count": "3",
            "dated_unit_count": "3",
            "undated_unit_count": "0",
            "date_only_count": "1",
            "datetime_count": "0",
            "year_month_count": "1",
            "year_only_count": "1",
            "date_coverage_percent": "100.00",
        },
        {
            "source_project": "Unknown",
            "unit_count": "1",
            "dated_unit_count": "0",
            "undated_unit_count": "1",
            "date_only_count": "0",
            "datetime_count": "0",
            "year_month_count": "0",
            "year_only_count": "0",
            "date_coverage_percent": "0.00",
        },
    ]


def test_source_date_precision_uses_highest_precision_per_unit():
    text = export_source_date_precision_csv(
        [
            unit(
                "a",
                "Source A",
                created_at=date(2024, 1, 2),
                metadata={"published_at": "2024-01-02T03:04:05Z"},
            ),
            unit(
                "b",
                "Source A",
                updated_at=datetime(2024, 1, 3, 4, 5, tzinfo=timezone.utc),
                metadata={"source_dates": ["2024", "2024-01"]},
            ),
            unit("c", "Source A", metadata={"created_at": "2024-01-02 03:04"}),
        ]
    )

    assert rows(text)[0]["datetime_count"] == "3"


def test_source_date_precision_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "date-precision.csv"
    units = [unit("a", "Source A", metadata={"date": "2024"})]

    expected = export_source_date_precision_csv(units)
    stats = export_source_date_precision_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "source_project_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_source_date_precision_is_deterministic_for_reversed_input():
    units = [
        unit("a", "Source B", metadata={"date": "2024"}),
        unit("b", "Source A", metadata={"date": "2024-01"}),
        unit("c", "Source A", metadata={"date": "2024-01-02"}),
    ]

    assert export_source_date_precision_csv(units) == export_source_date_precision_csv(reversed(units))
