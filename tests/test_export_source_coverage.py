from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_source_coverage_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


BASE_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    source_entity_type: str = "note",
    tags: list[str] | None = None,
    updated_at: datetime = BASE_TIME,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=tags or [],
        created_at=updated_at,
        ingested_at=updated_at,
        updated_at=updated_at,
    )


def test_source_coverage_groups_by_source_project_and_entity_type():
    older = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
    newest = datetime(2026, 5, 2, 12, 30, tzinfo=timezone.utc)

    report = export_source_coverage_markdown(
        [
            unit("unit-c", source_project=SourceProject.PINBOARD, tags=["reading"]),
            unit(
                "unit-a",
                source_project=SourceProject.MAX,
                source_entity_type="note",
                tags=["ai", "graph"],
                updated_at=older,
            ),
            unit(
                "unit-b",
                source_project=SourceProject.MAX,
                source_entity_type="note",
                tags=["ai"],
                updated_at=newest,
            ),
            unit(
                "unit-d",
                source_project=SourceProject.MAX,
                source_entity_type="task",
                tags=[],
                updated_at=BASE_TIME,
            ),
        ]
    )

    assert report == (
        "# Source Coverage Report\n"
        "\n"
        "## Summary\n"
        "\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        "| Units scanned | 4 |\n"
        "| Source groups | 3 |\n"
        "| Oldest updated | 2026-04-01T09:00:00+00:00 |\n"
        "| Newest updated | 2026-05-02T12:30:00+00:00 |\n"
        "\n"
        "## Source Groups\n"
        "\n"
        "| Source project | Entity type | Units | Oldest updated | Newest updated | Top tags |\n"
        "| --- | --- | ---: | --- | --- | --- |\n"
        "| max | note | 2 | 2026-04-01T09:00:00+00:00 | 2026-05-02T12:30:00+00:00 | ai (2), graph (1) |\n"
        "| max | task | 1 | 2026-05-01T10:15:00+00:00 | 2026-05-01T10:15:00+00:00 | _None_ |\n"
        "| pinboard | note | 1 | 2026-05-01T10:15:00+00:00 | 2026-05-01T10:15:00+00:00 | reading (1) |\n"
    )


def test_source_coverage_marks_stale_groups_when_threshold_is_supplied():
    report = export_source_coverage_markdown(
        [
            unit(
                "old",
                source_project=SourceProject.PINBOARD,
                source_entity_type="bookmark",
                updated_at=datetime(2026, 3, 15, 8, 0, tzinfo=timezone.utc),
            ),
            unit(
                "fresh",
                source_project=SourceProject.MAX,
                source_entity_type="note",
                updated_at=datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc),
            ),
        ],
        stale_after_days=30,
        as_of=datetime(2026, 5, 2, 0, 0, tzinfo=timezone.utc),
    )

    assert (
        "| Source project | Entity type | Units | Oldest updated | Newest updated | Top tags | Status |"
        in report
    )
    assert (
        "| max | note | 1 | 2026-05-01T08:00:00+00:00 | 2026-05-01T08:00:00+00:00 | _None_ | Current |"
        in report
    )
    assert (
        "| pinboard | bookmark | 1 | 2026-03-15T08:00:00+00:00 | 2026-03-15T08:00:00+00:00 | _None_ | Stale |"
        in report
    )
    assert "## Stale Sources" in report
    assert "- pinboard / bookmark: 1 units, newest update 2026-03-15T08:00:00+00:00" in report


def test_source_coverage_staleness_is_deterministic_without_as_of():
    report = export_source_coverage_markdown(
        [
            unit(
                "old",
                source_project=SourceProject.PINBOARD,
                updated_at=datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc),
            ),
            unit(
                "fresh",
                source_project=SourceProject.MAX,
                updated_at=datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc),
            ),
        ],
        stale_after_days=14,
    )

    assert (
        "| max | note | 1 | 2026-05-01T08:00:00+00:00 | 2026-05-01T08:00:00+00:00 | _None_ | Current |"
        in report
    )
    assert (
        "| pinboard | note | 1 | 2026-04-01T08:00:00+00:00 | 2026-04-01T08:00:00+00:00 | _None_ | Stale |"
        in report
    )


def test_source_coverage_limits_top_tags_and_escapes_markdown_table_cells():
    report = export_source_coverage_markdown(
        [
            unit("unit-a", source_project="custom|source", tags=["beta", "alpha", "alpha"]),
            unit("unit-b", source_project="custom|source", tags=["beta", "alpha", "gamma"]),
        ],
        top_tags_limit=2,
    )

    assert "| custom\\|source | note | 2 |" in report
    assert "alpha (2), beta (2)" in report
    assert "gamma" not in report


def test_source_coverage_empty_input_returns_zero_count_report():
    report = export_source_coverage_markdown([], stale_after_days=7)

    assert "| Units scanned | 0 |" in report
    assert "| Source groups | 0 |" in report
    assert "| _None_ | _None_ | 0 | _None_ | _None_ | _None_ | _None_ |" in report
    assert "_No stale source groups._" in report


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"stale_after_days": -1}, "stale_after_days"),
        ({"stale_after_days": True}, "stale_after_days"),
        ({"top_tags_limit": -1}, "top_tags_limit"),
        ({"top_tags_limit": False}, "top_tags_limit"),
    ],
)
def test_source_coverage_validates_options(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        export_source_coverage_markdown([], **kwargs)
