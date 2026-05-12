from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_source_tag_summary_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

TIME = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)


def unit(unit_id: str, source, tags):
    return KnowledgeUnit(
        id=unit_id,
        source_project=source,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=unit_id,
        content="content",
        content_type=ContentType.INSIGHT,
        tags=tags,
        created_at=TIME,
        ingested_at=TIME,
        updated_at=TIME,
    )


def test_source_tag_summary_groups_case_insensitive_counts_by_source():
    text = export_source_tag_summary_markdown(
        [
            unit("b", SourceProject.MAX, ["AI", "search"]),
            unit("a", SourceProject.MAX, ["ai"]),
            unit("c", SourceProject.KINDLE, ["quote"]),
        ]
    )

    assert "## kindle" in text
    assert "## max" in text
    assert "| AI | 2 |" in text
    assert "| search | 1 |" in text
    assert "| quote | 1 |" in text


def test_source_tag_summary_is_deterministic_and_limits_rows():
    units = [
        unit("c", SourceProject.MAX, ["beta"]),
        unit("a", SourceProject.MAX, ["alpha"]),
        unit("b", SourceProject.MAX, ["alpha"]),
        unit("d", SourceProject.KINDLE, ["beta"]),
    ]

    first = export_source_tag_summary_markdown(units, min_count=2, limit_per_source=1)
    second = export_source_tag_summary_markdown(reversed(units), min_count=2, limit_per_source=1)

    assert first == second
    assert "| alpha | 2 |" in first
    assert "| beta |" not in first.split("## max", 1)[1]


def test_source_tag_summary_writes_stats(tmp_path):
    path = tmp_path / "source-tags.md"
    stats = export_source_tag_summary_markdown([unit("a", SourceProject.MAX, ["ai"])], path)

    assert stats["units_scanned"] == 1
    assert stats["sources_exported"] == 1
    assert stats["tag_rows_exported"] == 1
    assert path.read_text(encoding="utf-8").startswith("# Source Tag Summary")


def test_source_tag_summary_validates_options():
    with pytest.raises(ValueError, match="min_count"):
        export_source_tag_summary_markdown([], min_count=0)
    with pytest.raises(ValueError, match="limit_per_source"):
        export_source_tag_summary_markdown([], limit_per_source=0)
