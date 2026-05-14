from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_source_date_gap_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    source_entity_type: str = "note",
    metadata: dict | None = None,
    created_at: object = UNIT_TIME,
    ingested_at: object = UNIT_TIME,
    updated_at: object = UNIT_TIME,
) -> KnowledgeUnit:
    item = KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )
    item.created_at = created_at
    item.ingested_at = ingested_at
    item.updated_at = updated_at
    return item


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_date_gap_csv_empty_input_returns_header():
    assert export_source_date_gap_csv([]) == (
        "source_project,source_entity_type,dated_unit_count,undated_unit_count,"
        "first_date,last_date,largest_gap_days,total_observations\n"
    )


def test_export_source_date_gap_csv_groups_with_unknown_fallbacks():
    text = export_source_date_gap_csv(
        [
            unit("b", source_project="", source_entity_type="", created_at="2026-01-10"),
            unit("a", source_project="alpha", source_entity_type="doc", created_at="2026-01-01"),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "alpha",
            "source_entity_type": "doc",
            "dated_unit_count": "1",
            "undated_unit_count": "0",
            "first_date": "2026-01-01",
            "last_date": "2026-05-01",
            "largest_gap_days": "120",
            "total_observations": "2",
        },
        {
            "source_project": "Unknown",
            "source_entity_type": "Unknown",
            "dated_unit_count": "1",
            "undated_unit_count": "0",
            "first_date": "2026-01-10",
            "last_date": "2026-05-01",
            "largest_gap_days": "111",
            "total_observations": "2",
        },
    ]


def test_export_source_date_gap_csv_extracts_core_and_metadata_dates():
    text = export_source_date_gap_csv(
        [
            unit(
                "a",
                metadata={
                    "published_at": "2026-01-04T05:00:00Z",
                    "dates": ["2026-01-10", datetime(2026, 1, 20, tzinfo=timezone.utc)],
                },
                created_at="2026-01-01",
                ingested_at="2026-01-02",
                updated_at="2026-01-03",
            ),
            unit(
                "b",
                metadata={"dates": ["not a date"]},
                created_at=None,
                ingested_at=None,
                updated_at=None,
            ),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "source_entity_type": "note",
            "dated_unit_count": "1",
            "undated_unit_count": "1",
            "first_date": "2026-01-01",
            "last_date": "2026-01-20",
            "largest_gap_days": "10",
            "total_observations": "6",
        }
    ]


def test_export_source_date_gap_csv_blanks_gap_for_one_observation():
    text = export_source_date_gap_csv([unit("a", created_at="2026-01-01", ingested_at=None, updated_at=None)])

    assert rows(text)[0]["largest_gap_days"] == ""


def test_export_source_date_gap_csv_path_mode_writes_same_content(tmp_path):
    units = [unit("a", created_at="2026-01-01")]
    expected = export_source_date_gap_csv(units)
    path = tmp_path / "date-gaps.csv"

    stats = export_source_date_gap_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
