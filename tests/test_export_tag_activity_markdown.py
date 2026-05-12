from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_tag_activity_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def dt(day: int, hour: int = 10) -> datetime:
    return datetime(2026, 5, day, hour, 0, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    tags: list[str] | None = None,
    source_project: SourceProject | str = SourceProject.MAX,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> KnowledgeUnit:
    created = created_at or dt(1)
    updated = updated_at or created
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
        created_at=created,
        ingested_at=created,
        updated_at=updated,
    )


def test_tag_activity_summarizes_counts_windows_and_sources():
    text = export_tag_activity_markdown(
        [
            unit("b", tags=["ai", "storage", "storage"], created_at=dt(3), updated_at=dt(4), source_project="pinboard"),
            unit("a", tags=[" ai ", "Research"], created_at=dt(1), updated_at=dt(2), source_project=SourceProject.MAX),
            unit("c", tags=["ai"], created_at=dt(2), updated_at=dt(6), source_project=SourceProject.MAX),
            unit("d", tags=[], created_at=dt(7), updated_at=dt(8), source_project="other"),
        ]
    )

    assert text == (
        "# Tag Activity Report\n"
        "\n"
        "## Summary\n"
        "\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        "| Units scanned | 4 |\n"
        "| Tags reported | 3 |\n"
        "| Min count | 1 |\n"
        "\n"
        "## Tags\n"
        "\n"
        "| Tag | Units | First created | Newest updated | Sources | Top sources |\n"
        "| --- | ---: | --- | --- | ---: | --- |\n"
        "| ai | 3 | 2026-05-01T10:00:00+00:00 | 2026-05-06T10:00:00+00:00 | 2 | max (2); pinboard (1) |\n"
        "| Research | 1 | 2026-05-01T10:00:00+00:00 | 2026-05-02T10:00:00+00:00 | 1 | max (1) |\n"
        "| storage | 1 | 2026-05-03T10:00:00+00:00 | 2026-05-04T10:00:00+00:00 | 1 | pinboard (1) |\n"
    )


def test_tag_activity_filters_by_min_count_and_orders_by_count_then_tag():
    text = export_tag_activity_markdown(
        [
            unit("a", tags=["beta", "alpha"], source_project="z"),
            unit("b", tags=["beta", "alpha"], source_project="a"),
            unit("c", tags=["beta"], source_project="a"),
            unit("d", tags=["sparse"], source_project="a"),
        ],
        min_count=2,
    )

    beta_row = "| beta | 3 |"
    alpha_row = "| alpha | 2 |"
    assert beta_row in text
    assert alpha_row in text
    assert text.index(beta_row) < text.index(alpha_row)
    assert "sparse" not in text
    assert "a (2); z (1)" in text


def test_tag_activity_empty_report_and_path_stats(tmp_path):
    path = tmp_path / "reports" / "tag-activity.md"

    stats = export_tag_activity_markdown([unit("a", tags=["one"])], path, min_count=2)

    assert stats == {
        "path": str(path),
        "units_scanned": 1,
        "tag_count": 0,
        "bytes_written": path.stat().st_size,
    }
    assert "| _None_ | 0 | _None_ | _None_ | 0 | _None_ |" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("min_count", [0, -1, True])
def test_tag_activity_validates_min_count(min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        export_tag_activity_markdown([], min_count=min_count)


def test_tag_activity_is_importable_from_graph_export():
    from graph.export import export_tag_activity_markdown as imported

    assert imported is export_tag_activity_markdown
