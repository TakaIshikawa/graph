from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.kindle_my_clippings import KindleMyClippingsAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_kindle_my_clippings_parses_multiple_blocks(tmp_path):
    path = tmp_path / "My Clippings.txt"
    path.write_text(
        """Book One (Ada Author)
- Your Highlight on page 12 | location 101-102 | Added on Tuesday, January 2, 2024 3:04:05 PM

Important passage.
==========
Article Without Author
- Your Note at location 55 | Added on Wednesday, January 3, 2024 04:05:06

Remember this.
==========
Book One (Ada Author)
- Your Bookmark on page 13 | location 120 | Added on Thursday, January 4, 2024 5:06:07 PM

==========
Book One (Ada Author)
- Your Highlight at location 130 | Added on Thursday, January 4, 2024 5:06:07 PM

==========
""",
        encoding="utf-8",
    )

    result = KindleMyClippingsAdapter(path=str(path)).ingest()

    assert len(result.units) == 3
    highlight = next(unit for unit in result.units if unit.metadata["clipping_type"] == "highlight")
    note = next(unit for unit in result.units if unit.metadata["clipping_type"] == "note")
    bookmark = next(unit for unit in result.units if unit.metadata["clipping_type"] == "bookmark")

    assert highlight.source_project == SourceProject.KINDLE
    assert highlight.source_entity_type == "clipping"
    assert highlight.title == "Book One: Highlight (Page 12 - Location 101-102)"
    assert highlight.content == "Important passage."
    assert highlight.content_type == ContentType.INSIGHT
    assert highlight.metadata == {
        "title": "Book One",
        "author": "Ada Author",
        "clipping_type": "highlight",
        "location": "101-102",
        "page": "12",
        "clipped_at": "Tuesday, January 2, 2024 3:04:05 PM",
        "source_file": str(path),
        "block_index": 0,
    }
    assert highlight.tags == ["Ada Author"]
    assert highlight.created_at == datetime(2024, 1, 2, 15, 4, 5, tzinfo=timezone.utc)

    assert note.title == "Article Without Author: Note (Location 55)"
    assert note.metadata["author"] == ""
    assert note.content == "Remember this."
    assert note.created_at == datetime(2024, 1, 3, 4, 5, 6, tzinfo=timezone.utc)

    assert bookmark.content_type == ContentType.METADATA
    assert bookmark.content == "Bookmark in Book One\nAuthor: Ada Author\nPage: 13\nLocation: 120"
    assert bookmark.metadata["block_index"] == 2


def test_kindle_my_clippings_source_ids_are_stable(tmp_path):
    path = tmp_path / "My Clippings.txt"
    path.write_text(
        """Stable Book (Author)
- Your Highlight at location 1 | Added on 2024-01-02

Stable text.
==========
""",
        encoding="utf-8",
    )

    first = KindleMyClippingsAdapter(path=str(path)).ingest().units[0]
    second = KindleMyClippingsAdapter(path=str(path)).ingest().units[0]

    assert first.source_id == second.source_id


def test_kindle_my_clippings_since_and_entity_filters(tmp_path):
    path = tmp_path / "My Clippings.txt"
    path.write_text(
        """Old Book
- Your Highlight at location 1 | Added on Monday, January 1, 2024 1:00:00 PM

Old.
==========
New Book
- Your Highlight at location 2 | Added on Monday, January 8, 2024 1:00:00 PM

New.
==========
""",
        encoding="utf-8",
    )

    since = SyncState(
        source_project="kindle",
        source_entity_type="clipping",
        last_sync_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    filtered = KindleMyClippingsAdapter(path=str(path)).ingest(since=since)
    wrong_entity = KindleMyClippingsAdapter(path=str(path)).ingest(entity_types=["book"])

    assert [unit.content for unit in filtered.units] == ["New."]
    assert wrong_entity.units == []
