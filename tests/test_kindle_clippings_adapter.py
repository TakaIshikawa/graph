"""Tests for the Kindle My Clippings.txt adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.kindle_clippings import KindleClippingsAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_kindle_clippings_parses_highlight_note_and_bookmark(tmp_path):
    clippings = tmp_path / "My Clippings.txt"
    clippings.write_text(
        """The Book Title (Ada Author)
- Your Highlight on page 7 | location 101-102 | Added on Tuesday, January 2, 2024 3:04:05 PM

Highlighted text.
==========
Article Without Author
- Your Note at location 55 | Added on Wednesday, January 3, 2024 04:05:06

Remember this point.
==========
The Book Title (Ada Author)
- Your Bookmark on page 8 | Added on Thursday, January 4, 2024 5:06:07 PM

==========
""",
        encoding="utf-8",
    )

    result = KindleClippingsAdapter(path=str(clippings)).ingest()

    assert len(result.units) == 3
    highlight = next(unit for unit in result.units if unit.metadata["clipping_type"] == "highlight")
    note = next(unit for unit in result.units if unit.metadata["clipping_type"] == "note")
    bookmark = next(unit for unit in result.units if unit.metadata["clipping_type"] == "bookmark")

    assert highlight.source_project == SourceProject.KINDLE
    assert highlight.source_entity_type == "clipping"
    assert highlight.title == "The Book Title: Highlight (Page 7 - Location 101-102)"
    assert highlight.content == "Highlighted text."
    assert highlight.content_type == ContentType.INSIGHT
    assert highlight.metadata == {
        "book_title": "The Book Title",
        "author": "Ada Author",
        "clipping_type": "highlight",
        "page": "7",
        "location": "101-102",
        "added_at": "Tuesday, January 2, 2024 3:04:05 PM",
        "source_file": "My Clippings.txt",
    }
    assert highlight.tags == ["Ada Author"]
    assert highlight.created_at == datetime(2024, 1, 2, 15, 4, 5, tzinfo=timezone.utc)

    assert note.title == "Article Without Author: Note (Location 55)"
    assert note.metadata["author"] == ""
    assert note.metadata["page"] == ""
    assert note.content == "Remember this point."

    assert bookmark.content_type == ContentType.METADATA
    assert bookmark.content == "The Book Title: Bookmark (Page 8)"


def test_kindle_clippings_skips_invalid_and_empty_non_bookmark_blocks(tmp_path):
    clippings = tmp_path / "My Clippings.txt"
    clippings.write_text(
        """Too short
==========
Book (Author)
- Your Highlight at location 10 | Added on Not a Date

==========
Book (Author)
- Your Highlight at location 11 | Added on Friday, January 5, 2024 6:07:08 PM

Valid highlight.
==========
""",
        encoding="utf-8",
    )

    result = KindleClippingsAdapter(path=str(clippings)).ingest()

    assert len(result.units) == 1
    assert result.units[0].content == "Valid highlight."


def test_kindle_clippings_since_filter_and_entity_filter(tmp_path):
    clippings = tmp_path / "My Clippings.txt"
    clippings.write_text(
        """Old Book (Author)
- Your Highlight at location 1 | Added on Monday, January 1, 2024 1:00:00 PM

Old.
==========
New Book (Author)
- Your Highlight at location 2 | Added on Monday, January 8, 2024 1:00:00 PM

New.
==========
""",
        encoding="utf-8",
    )

    filtered = KindleClippingsAdapter(path=str(clippings)).ingest(
        since=SyncState(
            source_project="kindle",
            source_entity_type="clipping",
            last_sync_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    )
    wrong_entity = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["book"])

    assert [unit.content for unit in filtered.units] == ["New."]
    assert wrong_entity.units == []
    assert wrong_entity.edges == []


def test_kindle_clippings_adapter_is_registered():
    assert "kindle_clippings" in list_adapters()
    adapter = get_adapter("kindle_clippings", path="/tmp/My Clippings.txt")
    assert isinstance(adapter, KindleClippingsAdapter)
    assert adapter.name == "kindle_clippings"
