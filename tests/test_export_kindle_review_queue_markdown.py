from __future__ import annotations

from datetime import datetime, timezone

from graph.export.kindle_review_queue_markdown import export_units_to_kindle_review_queue_markdown
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def _unit(source_id: str, *, book="Book", author="Author", location="1", created_at="2025-01-01T00:00:00Z", tags=None, clipping_type="highlight"):
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return KnowledgeUnit(
        source_project=SourceProject.KINDLE,
        source_id=source_id,
        source_entity_type="clipping",
        title=f"{book}: Highlight",
        content=f"Highlight {source_id}",
        metadata={"book_title": book, "author": author, "location": location, "clipping_type": clipping_type},
        tags=tags or ["review"],
        created_at=dt,
        updated_at=dt,
    )


def test_kindle_review_queue_groups_books_sorts_by_location_and_filters_tags():
    text = export_units_to_kindle_review_queue_markdown(
        [
            _unit("b", location="20", tags=["skip"]),
            _unit("a", location="3", tags=["review"]),
            _unit("note", location="1", clipping_type="note", tags=["review"]),
        ],
        tag_filter="review",
    )

    assert text == (
        "# Kindle Review Queue\n\n"
        "## Book - Author\n\n"
        "- Highlight a\n"
        "  Tags: review\n"
        "  Source: a\n"
    )


def test_kindle_review_queue_writes_path(tmp_path):
    path = tmp_path / "kindle.md"

    stats = export_units_to_kindle_review_queue_markdown([_unit("a")], path)

    assert stats == {"path": str(path), "bytes_written": path.stat().st_size}
    assert "## Book - Author" in path.read_text(encoding="utf-8")
