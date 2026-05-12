from __future__ import annotations

import pytest

from graph.export import export_metadata_date_histogram_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    metadata: dict,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="event",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def test_metadata_date_histogram_buckets_nested_dates_and_reports_skips():
    text = export_metadata_date_histogram_markdown(
        [
            unit("b", {"event": {"date": "2026-05-20T10:30:00Z"}}, source_project="pinboard"),
            unit("a", {"event": {"date": "2026-05-01"}}),
            unit("c", {"event": {"date": "2026-06-01"}}),
            unit("d", {"event": {"date": "not a date"}}),
            unit("e", {"event": {}}),
        ],
        "event.date",
    )

    assert "| Metadata path | event.date |" in text
    assert "| Valid units | 3 |" in text
    assert "| Missing units | 1 |" in text
    assert "| Invalid units | 1 |" in text
    assert "| Skipped units | 2 |" in text
    assert "| 2026-05 | 2 | max (1); pinboard (1) | Unit a; Unit b |" in text
    assert "| 2026-06 | 1 | max (1) | Unit c |" in text


def test_metadata_date_histogram_supports_day_week_year_and_source_filter():
    units = [
        unit("a", {"date": "2025-12-31"}, source_project=SourceProject.MAX),
        unit("b", {"date": "2026-01-01"}, source_project=SourceProject.MAX),
        unit("c", {"date": "2026-01-05"}, source_project=SourceProject.PRESENCE),
    ]

    day = export_metadata_date_histogram_markdown(units, "date", bucket="day", source_project="max")
    week = export_metadata_date_histogram_markdown(units, "date", bucket="week")
    year = export_metadata_date_histogram_markdown(units, "date", bucket="year")

    assert "| Source project | max |" in day
    assert "| Units scanned | 2 |" in day
    assert "| 2025-12-31 | 1 | max (1) | Unit a |" in day
    assert "| 2026-01-01 | 1 | max (1) | Unit b |" in day
    assert "Unit c" not in day
    assert "| 2025-12-29 | 2 | max (2) | Unit a; Unit b |" in week
    assert "| 2026-01-05 | 1 | presence (1) | Unit c |" in week
    assert "| 2026 | 2 | max (1); presence (1) | Unit b; Unit c |" in year


def test_metadata_date_histogram_writes_same_markdown(tmp_path):
    path = tmp_path / "reports" / "dates.md"
    units = [unit("a", {"date": "2026-05-01"})]

    text = export_metadata_date_histogram_markdown(units, "date", bucket="day")
    stats = export_metadata_date_histogram_markdown(units, "date", path, bucket="day")

    assert path.read_text(encoding="utf-8") == text
    assert stats == {
        "path": str(path),
        "metadata_path": "date",
        "bucket": "day",
        "source_project": None,
        "units_scanned": 1,
        "buckets_exported": 1,
        "missing_count": 0,
        "invalid_count": 0,
        "bytes_written": path.stat().st_size,
    }


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (([], ""), {}, "metadata_path must be a non-empty string"),
        (([], "date"), {"bucket": "quarter"}, "Unsupported metadata date histogram bucket"),
        (([], "date"), {"source_project": ""}, "source_project must be a non-empty string or None"),
    ],
)
def test_metadata_date_histogram_validates_options(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        export_metadata_date_histogram_markdown(*args, **kwargs)


def test_metadata_date_histogram_is_importable_from_graph_export():
    from graph.export import export_metadata_date_histogram_markdown as imported

    assert imported is export_metadata_date_histogram_markdown
