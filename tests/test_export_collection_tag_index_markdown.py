from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_collection_tag_index_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    title: str,
    tags: list[str] | None,
    source_project: SourceProject | str = SourceProject.MAX,
    source_entity_type: str = "note",
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=title,
        content="content",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def test_export_collection_tag_index_markdown_groups_units_by_tag_deterministically():
    units = [
        unit("b", title="Beta", tags=["research", "alpha"]),
        unit("a", title="Alpha", tags=["research"]),
    ]

    text = export_collection_tag_index_markdown(reversed(units))

    assert text == (
        "# Collection Tag Index\n"
        "\n"
        "## Summary\n"
        "\n"
        "| Tag | Units |\n"
        "| --- | ---: |\n"
        "| alpha | 1 |\n"
        "| research | 2 |\n"
        "\n"
        "## alpha\n"
        "\n"
        "| ID | Title | Source | Type | Updated |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| b | Beta | max | note | 2026-05-01T10:15:00+00:00 |\n"
        "\n"
        "## research\n"
        "\n"
        "| ID | Title | Source | Type | Updated |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| a | Alpha | max | note | 2026-05-01T10:15:00+00:00 |\n"
        "| b | Beta | max | note | 2026-05-01T10:15:00+00:00 |\n"
    )


def test_export_collection_tag_index_markdown_escapes_markdown_table_cells_and_headings():
    text = export_collection_tag_index_markdown(
        [
            unit(
                "unit|a",
                title="A | B",
                tags=["tag|one", "ref[1](x)"],
                source_project="src|docs",
                source_entity_type="note|draft",
            )
        ],
        title="Tags [Index]",
    )

    assert "# Tags \\[Index\\]\n" in text
    assert "| tag\\|one | 1 |" in text
    assert "## ref\\[1\\]\\(x\\)" in text
    assert "| unit\\|a | A \\| B | src\\|docs | note\\|draft | 2026-05-01T10:15:00+00:00 |" in text


def test_export_collection_tag_index_markdown_omits_untagged_by_default_and_can_include_them():
    units = [
        unit("a", title="Alpha", tags=["tagged"]),
        unit("b", title="Beta", tags=[]),
    ]

    default_text = export_collection_tag_index_markdown(units)
    included_text = export_collection_tag_index_markdown(units, include_untagged=True)

    assert "_untagged" not in default_text
    assert "| _untagged | 1 |" in included_text
    assert "## _untagged" in included_text
    assert "| b | Beta | max | note | 2026-05-01T10:15:00+00:00 |" in included_text


def test_export_collection_tag_index_markdown_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "tags.md"
    units = [unit("a", title="Alpha", tags=["tagged"])]

    expected = export_collection_tag_index_markdown(units)
    stats = export_collection_tag_index_markdown(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "tag_count": 1,
        "bytes_written": path.stat().st_size,
    }
