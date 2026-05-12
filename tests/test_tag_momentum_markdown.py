from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_tag_momentum_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, updated_at: datetime, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags,
        created_at=updated_at,
        ingested_at=updated_at,
        updated_at=updated_at,
    )


def dt(day: int) -> datetime:
    return datetime(2026, 5, day, 12, tzinfo=timezone.utc)


def test_tag_momentum_uses_newest_timestamp_and_window_boundaries():
    units = [
        unit("recent-anchor", dt(10), ["alpha", "beta"]),
        unit("recent-inside", dt(8), ["alpha"]),
        unit("previous-boundary-included", dt(5), ["alpha"]),
        unit("previous-inside", dt(3), ["gamma"]),
        unit("previous-boundary-excluded", dt(1), ["alpha"]),
    ]

    text = export_tag_momentum_markdown(units, recent_days=5, previous_days=4)

    assert "| Newest timestamp | 2026-05-10T12:00:00+00:00 |" in text
    assert "| Recent units | 2 |" in text
    assert "| Previous units | 2 |" in text
    assert "| alpha | 2 | 1 | +1 | +100.0% |" in text
    assert "| beta | 1 | 0 | +1 | _N/A_ |" in text
    assert "gamma" not in text


def test_tag_momentum_filters_escapes_and_orders_deterministically():
    units = [
        unit("b", dt(10), [r"path\\tag"]),
        unit("a", dt(10), ["beta", "alpha|pipe"]),
        unit("c", dt(9), ["beta", "alpha|pipe"]),
        unit("d", dt(8), ["rare"]),
        unit("e", dt(7), ["beta"]),
    ]

    first = export_tag_momentum_markdown(units, recent_days=4, min_recent_count=2)
    second = export_tag_momentum_markdown(reversed(units), recent_days=4, min_recent_count=2)

    assert first == second
    assert "| beta | 3 | 0 | +3 | _N/A_ |\n| alpha\\|pipe | 2 | 0 | +2 | _N/A_ |" in first
    assert r"path\\\\tag" not in first
    assert "rare" not in first


def test_tag_momentum_writes_path_and_returns_stats(tmp_path):
    output_path = tmp_path / "reports" / "momentum.md"
    units = [unit("a", dt(10), ["alpha"]), unit("b", dt(7), ["alpha"])]

    text = export_tag_momentum_markdown(units, recent_days=5, previous_days=3)
    stats = export_tag_momentum_markdown(units, output_path, recent_days=5, previous_days=3)

    assert output_path.read_text(encoding="utf-8") == text
    assert stats == {
        "path": str(output_path),
        "rows_exported": 1,
        "units_scanned": 2,
        "recent_unit_count": 2,
        "previous_unit_count": 0,
        "recent_days": 5,
        "previous_days": 3,
        "date_field": "updated_at",
        "min_recent_count": 1,
        "bytes_written": output_path.stat().st_size,
    }


def test_tag_momentum_supports_selected_date_field():
    selected = unit("a", dt(10), ["alpha"])
    old_updated = unit("b", dt(1), ["alpha"])
    old_updated.created_at = dt(9)

    text = export_tag_momentum_markdown([selected, old_updated], date_field="created_at", recent_days=2)

    assert "| Recent units | 2 |" in text
    assert "| alpha | 2 | 0 | +2 | _N/A_ |" in text


@pytest.mark.parametrize("recent_days", [0, -1, "2", None, True])
def test_tag_momentum_validates_recent_days(recent_days):
    with pytest.raises(ValueError, match="recent_days must be a positive integer"):
        export_tag_momentum_markdown([], recent_days=recent_days)


@pytest.mark.parametrize("previous_days", [0, -1, "2", None, True])
def test_tag_momentum_validates_previous_days(previous_days):
    with pytest.raises(ValueError, match="previous_days must be a positive integer"):
        export_tag_momentum_markdown([], previous_days=previous_days)


@pytest.mark.parametrize("min_recent_count", [0, -1, "2", None, True])
def test_tag_momentum_validates_min_recent_count(min_recent_count):
    with pytest.raises(ValueError, match="min_recent_count must be a positive integer"):
        export_tag_momentum_markdown([], min_recent_count=min_recent_count)


def test_tag_momentum_validates_date_field():
    with pytest.raises(ValueError, match="date_field must be one of"):
        export_tag_momentum_markdown([], date_field="metadata.date")


def test_tag_momentum_is_importable_from_graph_export():
    from graph.export import export_tag_momentum_markdown as imported

    assert imported is export_tag_momentum_markdown
