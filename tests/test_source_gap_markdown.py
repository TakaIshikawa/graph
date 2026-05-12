from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_source_gap_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    created_at: datetime,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    source_entity_type: str = "note",
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        created_at=created_at,
        ingested_at=created_at,
        updated_at=created_at,
    )


def dt(month: int, day: int) -> datetime:
    return datetime(2026, month, day, 12, tzinfo=timezone.utc)


def test_source_gap_detects_strict_gaps_by_group_and_sorts_largest_first():
    units = [
        unit("a", dt(1, 1)),
        unit("b", dt(1, 31)),
        unit("c", dt(3, 5)),
        unit("d", dt(5, 10), source_entity_type="task"),
        unit("e", dt(6, 25), source_entity_type="task"),
        unit("f", dt(4, 1), source_project="pinboard"),
        unit("g", dt(5, 20), source_project="pinboard"),
    ]

    text = export_source_gap_markdown(units, gap_days=30)

    assert "| Units scanned | 7 |" in text
    assert "| Gaps reported | 3 |" in text
    assert (
        "| max | task | 2026-05-10T12:00:00+00:00 | 2026-06-25T12:00:00+00:00 | 46.0 |\n"
        "| pinboard | note | 2026-04-01T12:00:00+00:00 | 2026-05-20T12:00:00+00:00 | 49.0 |"
    ) not in text
    assert "| pinboard | note | 2026-04-01T12:00:00+00:00 | 2026-05-20T12:00:00+00:00 | 49.0 |" in text
    assert "| max | task | 2026-05-10T12:00:00+00:00 | 2026-06-25T12:00:00+00:00 | 46.0 |" in text
    assert "| max | note | 2026-01-31T12:00:00+00:00 | 2026-03-05T12:00:00+00:00 | 33.0 |" in text
    assert "2026-01-01T12:00:00+00:00 | 2026-01-31T12:00:00+00:00" not in text


def test_source_gap_limits_results_and_is_deterministic():
    units = [
        unit("c", dt(4, 1)),
        unit("a", dt(1, 1)),
        unit("b", dt(2, 15)),
        unit("d", dt(6, 1)),
    ]

    first = export_source_gap_markdown(units, gap_days=30, max_gaps=1)
    second = export_source_gap_markdown(reversed(units), gap_days=30, max_gaps=1)

    assert first == second
    assert "| Gaps reported | 1 |" in first
    assert "| max | note | 2026-04-01T12:00:00+00:00 | 2026-06-01T12:00:00+00:00 | 61.0 |" in first
    assert "2026-01-01T12:00:00+00:00 | 2026-02-15T12:00:00+00:00" not in first


def test_source_gap_supports_selected_date_field_and_escapes_cells():
    first = unit("a", dt(1, 1), source_project="source|one", source_entity_type=r"type\\name")
    second = unit("b", dt(1, 2), source_project="source|one", source_entity_type=r"type\\name")
    first.updated_at = dt(2, 1)
    second.updated_at = dt(3, 10)

    text = export_source_gap_markdown([first, second], date_field="updated_at", gap_days=30)

    assert "| Date field | updated_at |" in text
    assert r"| source\|one | type\\\\name | 2026-02-01T12:00:00+00:00 | 2026-03-10T12:00:00+00:00 | 37.0 |" in text


def test_source_gap_empty_and_no_gap_reports():
    empty = export_source_gap_markdown([])
    no_gap = export_source_gap_markdown([unit("a", dt(1, 1)), unit("b", dt(1, 20))])

    assert "| Units scanned | 0 |" in empty
    assert "| _None_ | _None_ | _None_ | _None_ | 0.0 |" in empty
    assert "| Gaps reported | 0 |" in no_gap
    assert "| _None_ | _None_ | _None_ | _None_ | 0.0 |" in no_gap


def test_source_gap_writes_path_and_returns_stats(tmp_path):
    output_path = tmp_path / "reports" / "gaps.md"
    units = [unit("a", dt(1, 1)), unit("b", dt(2, 15))]

    text = export_source_gap_markdown(units, gap_days=30)
    stats = export_source_gap_markdown(units, output_path, gap_days=30)

    assert output_path.read_text(encoding="utf-8") == text
    assert stats == {
        "path": str(output_path),
        "units_scanned": 2,
        "gaps_exported": 1,
        "date_field": "created_at",
        "gap_days": 30,
        "max_gaps": 10,
        "bytes_written": output_path.stat().st_size,
    }


def test_source_gap_validates_date_field():
    with pytest.raises(ValueError, match="date_field must be one of"):
        export_source_gap_markdown([], date_field="metadata.date")


@pytest.mark.parametrize("gap_days", [0, -1, "2", None, True])
def test_source_gap_validates_gap_days(gap_days):
    with pytest.raises(ValueError, match="gap_days must be a positive integer"):
        export_source_gap_markdown([], gap_days=gap_days)


@pytest.mark.parametrize("max_gaps", [0, -1, "2", None, True])
def test_source_gap_validates_max_gaps(max_gaps):
    with pytest.raises(ValueError, match="max_gaps must be a positive integer"):
        export_source_gap_markdown([], max_gaps=max_gaps)


def test_source_gap_is_importable_from_graph_export():
    from graph.export import export_source_gap_markdown as imported

    assert imported is export_source_gap_markdown
