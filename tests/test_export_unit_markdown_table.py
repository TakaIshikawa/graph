from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_units_to_markdown_table
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    content_type: ContentType = ContentType.FINDING,
    title: str | None = None,
    content: str = "Alpha content",
    tags: list[str] | None = None,
    confidence: float | None = 0.8,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title if title is not None else f"Title {unit_id}",
        content=content,
        content_type=content_type,
        tags=tags or ["storage", "solar"],
        confidence=confidence,
        utility_score=0.6,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def test_export_units_markdown_table_renders_default_columns_deterministically():
    text = export_units_to_markdown_table(
        [
            unit("unit-b", source_project=SourceProject.PINBOARD, content="Beta content"),
            unit("unit-a", content_type=ContentType.IDEA, content="Alpha content"),
        ]
    )

    assert text == (
        "| id | title | source_project | content_type | created_at | tags | content_excerpt |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| unit-a | Title unit-a | max | idea | 2026-05-01T10:15:00+00:00 | solar; storage | Alpha content |\n"
        "| unit-b | Title unit-b | pinboard | finding | 2026-05-01T10:15:00+00:00 | solar; storage | Beta content |\n"
    )


def test_export_units_markdown_table_supports_custom_fields_without_changing_defaults():
    custom = export_units_to_markdown_table(
        [unit("unit-a", confidence=None)],
        fields=["id", "confidence", "source_id"],
    )
    default = export_units_to_markdown_table([unit("unit-a", confidence=None)])

    assert custom == (
        "| id | confidence | source_id |\n"
        "| --- | --- | --- |\n"
        "| unit-a |  | source-unit-a |\n"
    )
    assert default.startswith(
        "| id | title | source_project | content_type | created_at | tags | content_excerpt |"
    )


def test_export_units_markdown_table_escapes_pipes_and_normalizes_multiline_values():
    text = export_units_to_markdown_table(
        [
            unit(
                "unit|a",
                title="Title | A\nsecond line",
                content="Line one\nLine | two",
                tags=["zeta | pipe", "alpha\nbeta"],
            )
        ]
    )

    assert "unit\\|a" in text
    assert "Title \\| A second line" in text
    assert "Line one Line \\| two" in text
    assert "alpha beta; zeta \\| pipe" in text


def test_export_units_markdown_table_bounds_content_excerpt():
    text = export_units_to_markdown_table([unit("unit-a", content="x" * 140)])

    assert f"| {'x' * 117}... |" in text
    assert "x" * 140 not in text


def test_export_units_markdown_table_optionally_writes_to_file(tmp_path):
    path = tmp_path / "reports" / "units.md"

    text = export_units_to_markdown_table([unit("unit-a")], path)

    assert path.read_text(encoding="utf-8") == text


def test_export_units_markdown_table_is_importable_from_graph_export():
    from graph.export import export_units_to_markdown_table as imported

    assert imported is export_units_to_markdown_table
