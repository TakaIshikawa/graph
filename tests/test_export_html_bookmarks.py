from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_units_to_bookmarks_html
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    title: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.BOOKMARKS_HTML,
        source_id=f"source-{unit_id}",
        source_entity_type="bookmark",
        title=title or f"Title {unit_id}",
        content="Bookmark content",
        content_type=ContentType.ARTIFACT,
        tags=tags or [],
        metadata=metadata or {},
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def test_export_units_to_bookmarks_html_emits_netscape_structure():
    text = export_units_to_bookmarks_html(
        [
            unit(
                "unit-a",
                title="Example & docs",
                tags=["docs", "graph"],
                metadata={"source_url": "https://example.test/?a=1&b=2", "add_date": 1777616100},
            )
        ]
    )

    assert text.startswith("<!DOCTYPE NETSCAPE-Bookmark-file-1>\n")
    assert "<DL><p>\n" in text
    assert (
        '<DT><A HREF="https://example.test/?a=1&amp;b=2" ADD_DATE="1777616100" '
        'TAGS="docs,graph">Example &amp; docs</A>'
    ) in text


def test_export_units_to_bookmarks_html_skips_units_without_url():
    text = export_units_to_bookmarks_html(
        [
            unit("unit-a", metadata={"source_url": "https://example.test/a"}),
            unit("unit-b", metadata={}),
        ]
    )

    assert "Title unit-a" in text
    assert "Title unit-b" not in text


def test_export_units_to_bookmarks_html_escapes_titles_tags_and_dates():
    text = export_units_to_bookmarks_html(
        [
            unit(
                "unit-a",
                title="<script>Title</script>",
                tags=['quote"tag', "alpha&beta"],
                metadata={"external_url": "https://example.test/a", "created_at": UNIT_TIME},
            )
        ]
    )

    assert "<script>" not in text
    assert "&lt;script&gt;Title&lt;/script&gt;" in text
    assert 'TAGS="alpha&amp;beta,quote&quot;tag"' in text
    assert 'ADD_DATE="1777630500"' in text


def test_export_units_to_bookmarks_html_writes_file_and_creates_parent(tmp_path):
    path = tmp_path / "nested" / "bookmarks.html"

    stats = export_units_to_bookmarks_html(
        [
            unit("unit-a", metadata={"url": "https://example.test/a"}),
            unit("unit-b"),
        ],
        path,
    )

    assert path.exists()
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "skipped_count": 1,
        "bytes_written": path.stat().st_size,
    }
