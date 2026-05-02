from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_tag_glossary_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    title: str | None = None,
    source_project: SourceProject | str = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
    tags: list[str] | None = None,
    updated_at: datetime | None = None,
) -> KnowledgeUnit:
    timestamp = updated_at or datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title if title is not None else f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=content_type,
        tags=tags or [],
        created_at=timestamp,
        ingested_at=timestamp,
        updated_at=timestamp,
    )


def test_export_tag_glossary_groups_counts_and_examples(tmp_path):
    path = tmp_path / "reports" / "tags.md"
    older = datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    newer = datetime(2026, 5, 2, 9, tzinfo=timezone.utc)

    stats = export_tag_glossary_markdown(
        [
            unit(
                "unit-b",
                title="Battery note",
                source_project=SourceProject.PINBOARD,
                content_type=ContentType.FINDING,
                tags=["storage", "solar"],
                updated_at=newer,
            ),
            unit(
                "unit-a",
                title="Solar note",
                source_project=SourceProject.MAX,
                content_type=ContentType.INSIGHT,
                tags=["storage", "storage"],
                updated_at=older,
            ),
        ],
        path,
    )

    text = path.read_text(encoding="utf-8")

    assert stats == {
        "path": str(path),
        "units_scanned": 2,
        "tags_scanned": 2,
        "tags_exported": 2,
        "min_count": 1,
        "include_examples": 3,
        "bytes_written": path.stat().st_size,
    }
    assert "## storage\n\n- Usage count: 2" in text
    assert "| max | 1 |" in text
    assert "| pinboard | 1 |" in text
    assert "| finding | 1 |" in text
    assert "| insight | 1 |" in text
    assert text.index("Battery note (`unit-b`)") < text.index("Solar note (`unit-a`)")


def test_export_tag_glossary_filters_by_min_count(tmp_path):
    path = tmp_path / "tags.md"

    export_tag_glossary_markdown(
        [
            unit("unit-a", tags=["keep", "drop"]),
            unit("unit-b", tags=["keep"]),
        ],
        path,
        min_count=2,
    )

    text = path.read_text(encoding="utf-8")

    assert "## keep" in text
    assert "## drop" not in text
    assert "| Tags scanned | 2 |" in text
    assert "| Tags exported | 1 |" in text


def test_export_tag_glossary_is_deterministic_and_sanitizes_markdown(tmp_path):
    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"
    units = [
        unit("unit-b", title="Title [B](x)", tags=["zeta | pipe", "topic #1"]),
        unit("unit-a", title="Title A", tags=["topic\n#1"]),
    ]

    export_tag_glossary_markdown(units, first_path)
    export_tag_glossary_markdown(reversed(units), second_path)

    first = first_path.read_text(encoding="utf-8")

    assert first == second_path.read_text(encoding="utf-8")
    assert "## topic \\#1" in first
    assert "## zeta | pipe" in first
    assert "Title \\[B\\]\\(x\\) (`unit-b`)" in first
    assert "| Tags scanned | 2 |" in first


def test_export_tag_glossary_handles_empty_tags_and_no_matches(tmp_path):
    path = tmp_path / "tags.md"

    export_tag_glossary_markdown(
        [unit("unit-a", tags=["", "   "]), unit("unit-b", tags=[])],
        path,
    )

    text = path.read_text(encoding="utf-8")

    assert "| Units scanned | 2 |" in text
    assert "| Tags scanned | 0 |" in text
    assert "_No tags matched the export criteria._" in text


def test_export_tag_glossary_respects_include_examples(tmp_path):
    path = tmp_path / "tags.md"

    export_tag_glossary_markdown(
        [
            unit("unit-a", tags=["storage"]),
            unit("unit-b", tags=["storage"]),
        ],
        path,
        include_examples=1,
    )

    text = path.read_text(encoding="utf-8")

    assert "Title unit-a (`unit-a`)" in text
    assert "Title unit-b (`unit-b`)" not in text


@pytest.mark.parametrize("min_count", [0, -1, "2", True])
def test_export_tag_glossary_validates_min_count(tmp_path, min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        export_tag_glossary_markdown([], tmp_path / "tags.md", min_count=min_count)


@pytest.mark.parametrize("include_examples", [-1, "2", True])
def test_export_tag_glossary_validates_include_examples(tmp_path, include_examples):
    with pytest.raises(ValueError, match="include_examples must be a non-negative integer"):
        export_tag_glossary_markdown(
            [],
            tmp_path / "tags.md",
            include_examples=include_examples,
        )


def test_export_tag_glossary_is_importable_from_graph_export():
    from graph.export import export_tag_glossary_markdown as imported

    assert imported is export_tag_glossary_markdown
