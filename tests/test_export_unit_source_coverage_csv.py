from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from io import StringIO

from graph.export.unit_source_coverage_csv import export_units_to_source_coverage_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    source_project: SourceProject | str | None,
    source_id: str | None,
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
    created_at: object = "2026-05-01",
    ingested_at: object = "2026-05-02",
    updated_at: object = "2026-05-03",
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=source_id,
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
        created_at=created_at,
        ingested_at=ingested_at,
        updated_at=updated_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_source_coverage_empty_input_has_header_only():
    assert export_units_to_source_coverage_csv([]) == (
        "source_project,unit_count,unique_source_id_count,missing_source_id_count,"
        "url_coverage_percent,tag_coverage_percent,earliest_unit_date,latest_unit_date,"
        "duplicate_source_id_count\n"
    )


def test_unit_source_coverage_groups_by_source_project_and_counts_coverage():
    text = export_units_to_source_coverage_csv(
        [
            unit(
                "a",
                SourceProject.MAX,
                "max-1",
                metadata={"source_url": "https://example.com/a"},
                tags=["alpha"],
                created_at=datetime(2026, 5, 1, 23, 30, tzinfo=timezone.utc),
                ingested_at=date(2026, 5, 2),
                updated_at="2026-05-03T10:00:00Z",
            ),
            unit("b", SourceProject.MAX, "max-1", metadata={"url": ""}, tags=[]),
            unit("c", SourceProject.MAX, "  ", metadata={"links": ["https://example.com/c"]}, tags=["  "]),
            unit("d", "Other", "other-1", metadata={"link": "not a url"}, tags=["tag"]),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "unit_count": "3",
            "unique_source_id_count": "1",
            "missing_source_id_count": "1",
            "url_coverage_percent": "66.67",
            "tag_coverage_percent": "33.33",
            "earliest_unit_date": "2026-05-01",
            "latest_unit_date": "2026-05-03",
            "duplicate_source_id_count": "1",
        },
        {
            "source_project": "Other",
            "unit_count": "1",
            "unique_source_id_count": "1",
            "missing_source_id_count": "0",
            "url_coverage_percent": "0.00",
            "tag_coverage_percent": "100.00",
            "earliest_unit_date": "2026-05-01",
            "latest_unit_date": "2026-05-03",
            "duplicate_source_id_count": "0",
        },
    ]


def test_unit_source_coverage_includes_unknown_source_bucket():
    text = export_units_to_source_coverage_csv(
        [
            unit("a", None, "unknown-1", metadata={"url": "https://example.com"}, tags=["tag"]),
            unit("b", " \n ", "", metadata={}, tags=[]),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Unknown",
            "unit_count": "2",
            "unique_source_id_count": "1",
            "missing_source_id_count": "1",
            "url_coverage_percent": "50.00",
            "tag_coverage_percent": "50.00",
            "earliest_unit_date": "2026-05-01",
            "latest_unit_date": "2026-05-03",
            "duplicate_source_id_count": "0",
        }
    ]


def test_unit_source_coverage_ignores_invalid_dates():
    text = export_units_to_source_coverage_csv(
        [
            unit("a", "Source A", "a", created_at="bad", ingested_at=None, updated_at="2026-05-10"),
            unit("b", "Source A", "b", created_at="also bad", ingested_at="bad", updated_at=None),
        ]
    )

    assert rows(text)[0]["earliest_unit_date"] == "2026-05-10"
    assert rows(text)[0]["latest_unit_date"] == "2026-05-10"


def test_unit_source_coverage_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "unit-source-coverage.csv"
    units = [unit("a", "Source A", "a")]

    expected = export_units_to_source_coverage_csv(units)
    stats = export_units_to_source_coverage_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "source_project_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_unit_source_coverage_is_deterministic_for_reversed_input():
    units = [
        unit("a", "Source B", "b"),
        unit("b", "Source A", "a-1"),
        unit("c", "Source A", "a-2"),
    ]

    assert export_units_to_source_coverage_csv(units) == export_units_to_source_coverage_csv(
        reversed(units)
    )
