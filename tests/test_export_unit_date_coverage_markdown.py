from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_unit_date_coverage_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    source_project: SourceProject | str | None,
    *,
    created_at: object,
    ingested_at: object,
    updated_at: object,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={},
        created_at=created_at,
        ingested_at=ingested_at,
        updated_at=updated_at,
    )


def test_unit_date_coverage_reports_overall_and_by_source():
    early = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    late = datetime(2026, 1, 3, 8, 0, tzinfo=timezone.utc)
    markdown = export_unit_date_coverage_markdown(
        [
            unit("b", SourceProject.MAX, created_at=late, ingested_at=None, updated_at=late),
            unit("a", None, created_at=early, ingested_at=early, updated_at=None),
        ]
    )

    assert markdown.startswith("# Unit Date Coverage\n")
    assert "## Overall" in markdown
    assert "## max" in markdown
    assert "## Unknown" in markdown
    assert (
        "| created_at | 2 | 0 | 100.00 | "
        "2026-01-01T08:00:00+00:00 | 2026-01-03T08:00:00+00:00 |"
    ) in markdown
    assert "| ingested_at | 1 | 1 | 50.00 | 2026-01-01T08:00:00+00:00 | 2026-01-01T08:00:00+00:00 |" in markdown


def test_unit_date_coverage_renders_string_like_values_safely():
    markdown = export_unit_date_coverage_markdown(
        [
            unit(
                "a",
                "Docs | Team",
                created_at="2026-01-02 | draft",
                ingested_at=" 2026-01-01\n10:00 ",
                updated_at=None,
            )
        ]
    )

    assert "## Docs \\| Team" in markdown
    assert "| created_at | 1 | 0 | 100.00 | 2026-01-02 \\| draft | 2026-01-02 \\| draft |" in markdown
    assert "| ingested_at | 1 | 0 | 100.00 | 2026-01-01 10:00 | 2026-01-01 10:00 |" in markdown
    assert "| updated_at | 0 | 1 | 0.00 |  |  |" in markdown


def test_unit_date_coverage_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "date-coverage.md"
    units = [
        unit(
            "a",
            "Source A",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ingested_at=None,
            updated_at=None,
        )
    ]

    expected = export_unit_date_coverage_markdown(units)
    stats = export_unit_date_coverage_markdown(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "source_project_count": 1,
        "bytes_written": path.stat().st_size,
    }
