from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_date_range_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, created_at=None, source_project: str = "Project") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
        created_at=created_at,
        updated_at=None,
        ingested_at=None,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_date_range_csv_empty_input_has_header_only():
    assert export_unit_date_range_csv([]) == (
        "source,unit_id,title,date_count,earliest_date,latest_date,span_days,date_keys\n"
    )


def test_unit_date_range_csv_parses_iso_dates_and_ignores_unparseable_values():
    text = export_unit_date_range_csv(
        [
            unit(
                "a",
                metadata={"date": ["2026-05-10", "not a date"], "published_at": "2026-05-12T08:30:00+00:00"},
                created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            ),
            unit("b", metadata={"date": "never"}),
        ]
    )

    assert rows(text) == [
        {
            "source": "Project",
            "unit_id": "a",
            "title": "Title a",
            "date_count": "3",
            "earliest_date": "2026-05-01",
            "latest_date": "2026-05-12",
            "span_days": "11",
            "date_keys": "created_at; date; published_at",
        }
    ]


def test_unit_date_range_csv_lists_repeated_metadata_key_once():
    text = export_unit_date_range_csv([unit("a", metadata={"date": ["2026-01-01", "2026-01-03"]})])

    assert rows(text)[0]["date_keys"] == "date"
    assert rows(text)[0]["date_count"] == "2"


def test_unit_date_range_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "dates.csv"
    units = [unit("a", metadata={"date": "2026-01-01"})]

    expected = export_unit_date_range_csv(units)
    stats = export_unit_date_range_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {"path": str(path), "unit_count": 1, "rows_exported": 1, "bytes_written": path.stat().st_size}
