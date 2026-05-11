from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_units_to_markdown_timeline
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

CREATED_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
INGESTED_TIME = datetime(2026, 5, 2, 8, 30, tzinfo=timezone.utc)
UPDATED_TIME = datetime(2026, 5, 3, 9, 45, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str | None = None,
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
    source_project: SourceProject | str = SourceProject.CSV,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title or f"Title {unit_id}",
        content="A compact research note.",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags if tags is not None else ["timeline", "export"],
        created_at=CREATED_TIME,
        ingested_at=INGESTED_TIME,
        updated_at=UPDATED_TIME,
    )


def undated_unit(unit_id: str) -> KnowledgeUnit:
    return unit(unit_id, metadata={"date": "not a date"}).model_copy(
        update={"created_at": None, "updated_at": None, "ingested_at": None}
    )


def test_export_units_to_markdown_timeline_groups_by_date_chronologically():
    text = export_units_to_markdown_timeline(
        [
            unit("b", "Beta", metadata={"date": "2026-01-02"}),
            unit("a", "Alpha", metadata={"date": "2026-01-01"}),
            unit("c", "Gamma", metadata={"date": "2026-01-02"}),
        ]
    )

    assert text.startswith("# Timeline\n\n## 2026-01-01\n\n- **Alpha**")
    assert text.index("## 2026-01-01") < text.index("## 2026-01-02")
    assert text.index("- **Beta**") < text.index("- **Gamma**")


def test_export_units_to_markdown_timeline_skips_undated_units_for_path_writes(tmp_path):
    path = tmp_path / "timeline.md"

    stats = export_units_to_markdown_timeline([unit("dated", metadata={"date": "2026-01-01"}), undated_unit("skip")], path)

    assert "Title dated" in path.read_text(encoding="utf-8")
    assert "Title skip" not in path.read_text(encoding="utf-8")
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "skipped_count": 1,
        "bytes_written": path.stat().st_size,
    }


def test_export_units_to_markdown_timeline_renders_source_url_links():
    text = export_units_to_markdown_timeline(
        unit("a", metadata={"date": "2026-01-01", "source_url": "https://example.test/a"})
    )

    assert "[source](https://example.test/a)" in text


def test_export_units_to_markdown_timeline_is_deterministic_and_handles_empty_metadata():
    units = [
        unit("b", "Same", metadata={"date": "2026-01-01"}, tags=[]),
        unit("a", "Same", metadata={"date": "2026-01-01"}, tags=["zeta", "", "alpha"]),
    ]

    text_a = export_units_to_markdown_timeline(units)
    text_b = export_units_to_markdown_timeline(list(reversed(units)))

    assert text_a == text_b
    assert "csv | tags: alpha, zeta" in text_a
    assert "- **Same**\n  - csv\n" in text_a


def test_export_units_to_markdown_timeline_writes_nested_paths(tmp_path):
    path = tmp_path / "nested" / "timeline.md"

    stats = export_units_to_markdown_timeline(unit("a", metadata={"date": "2026-01-01"}), path)

    assert path.read_text(encoding="utf-8").startswith("# Timeline")
    assert stats["path"] == str(path)
    assert stats["unit_count"] == 1
