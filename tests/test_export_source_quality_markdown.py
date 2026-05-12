from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_source_quality_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

OLD = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)
NEW = datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    source,
    *,
    title: str = "Title",
    content: str = "content",
    metadata: dict | None = None,
    tags: list[str] | None = None,
    updated_at: datetime = OLD,
):
    return KnowledgeUnit(
        id=unit_id,
        source_project=source,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
        created_at=OLD,
        ingested_at=OLD,
        updated_at=updated_at,
    )


def test_source_quality_groups_by_source_and_reports_missing_counts():
    text = export_source_quality_markdown(
        [
            unit("a", SourceProject.MAX, content="abcd", metadata={"x": 1}, tags=["ai"], updated_at=OLD),
            unit("b", SourceProject.MAX, title="", content="", updated_at=NEW),
            unit("c", "", content="abcdef", metadata={"y": 2}, tags=["search"], updated_at=OLD),
        ]
    )

    assert "| Units scanned | 3 |" in text
    assert "| Source projects | 2 |" in text
    assert "| Missing titles | 1 |" in text
    assert "| Missing content | 1 |" in text
    assert "| Missing metadata | 1 |" in text
    assert "| Missing tags | 1 |" in text
    assert "| Unknown | 1 | 0 | 0 | 0 | 0 | 6 | 2026-05-01T10:00:00+00:00 |" in text
    assert "| max | 2 | 1 | 1 | 1 | 1 | 2 | 2026-05-02T10:00:00+00:00 |" in text


def test_source_quality_is_deterministic_and_writes_stats(tmp_path):
    units = [
        unit("b", SourceProject.MAX, title="", content="", updated_at=NEW),
        unit("a", SourceProject.MAX, content="abcd", metadata={"x": 1}, tags=["ai"], updated_at=OLD),
    ]
    path = tmp_path / "quality.md"

    first = export_source_quality_markdown(units)
    second = export_source_quality_markdown(reversed(units))
    stats = export_source_quality_markdown(units, path)

    assert first == second
    assert stats["unit_count"] == 2
    assert stats["source_count"] == 1
    assert stats["bytes_written"] == path.stat().st_size
