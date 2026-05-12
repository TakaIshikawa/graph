"""Tests for the Kindle My Clippings.txt adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.kindle_clippings import KindleClippingsAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, EdgeRelation, SourceProject
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

    assert len(result.units) == 6
    assert len(result.edges) == 4
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

    book = next(unit for unit in result.units if unit.source_entity_type == "book" and unit.metadata["author"] == "Ada Author")
    assert book.metadata["book_title"] == "The Book Title"
    assert book.metadata["author"] == "Ada Author"
    assert book.metadata["clipping_count"] == 2
    assert book.metadata["highlight_count"] == 1
    assert book.metadata["note_count"] == 0
    assert book.metadata["bookmark_count"] == 1
    assert book.metadata["first_clipped_at"] == "2024-01-02T15:04:05+00:00"
    assert book.metadata["last_clipped_at"] == "2024-01-04T17:06:07+00:00"
    assert book.metadata["page_start"] == 7
    assert book.metadata["page_end"] == 8
    assert book.metadata["location_start"] == 101
    assert book.metadata["location_end"] == 101
    assert book.metadata["source_files"] == ["My Clippings.txt"]
    edge = next(edge for edge in result.edges if edge.to_unit_id == highlight.source_id)
    assert edge.from_unit_id == book.source_id
    assert edge.relation == EdgeRelation.CONTAINS
    author = next(unit for unit in result.units if unit.source_entity_type == "author")
    assert author.metadata["author"] == "Ada Author"
    assert author.metadata["book_count"] == 1
    assert author.metadata["clipping_count"] == 2
    assert author.metadata["highlight_count"] == 1
    assert author.metadata["bookmark_count"] == 1
    assert author.metadata["book_source_ids"] == [book.source_id]
    author_edge = next(edge for edge in result.edges if edge.from_unit_id == author.source_id)
    assert author_edge.to_unit_id == book.source_id
    assert author_edge.metadata["relation_type"] == "author_contains_book"


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

    clippings = [unit for unit in result.units if unit.source_entity_type == "clipping"]
    assert len(clippings) == 1
    assert clippings[0].content == "Valid highlight."


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
        ),
        entity_types=["clipping"],
    )
    books = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["book"])
    authors = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["author"])
    clipping_only = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["clipping"])
    wrong_entity = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["bookcase"])

    assert [unit.content for unit in filtered.units] == ["New."]
    assert {unit.source_entity_type for unit in books.units} == {"book"}
    assert books.edges == []
    assert {unit.source_entity_type for unit in authors.units} == {"author"}
    assert authors.edges == []
    assert {unit.source_entity_type for unit in clipping_only.units} == {"clipping"}
    assert clipping_only.edges == []
    assert wrong_entity.units == []
    assert wrong_entity.edges == []


def test_kindle_clippings_book_aggregate_summarizes_counts_and_locations(tmp_path):
    clippings = tmp_path / "My Clippings.txt"
    clippings.write_text(
        """Shared Book (Author)
- Your Highlight at location 10-12 | Added on Monday, January 1, 2024 1:00:00 PM

Highlight.
==========
Shared Book (Author)
- Your Note at location 30 | Added on Monday, January 1, 2024 2:00:00 PM

Note.
==========
""",
        encoding="utf-8",
    )

    book = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["book"]).units[0]

    assert book.metadata["clipping_count"] == 2
    assert book.metadata["highlight_count"] == 1
    assert book.metadata["note_count"] == 1
    assert book.metadata["location_start"] == 10
    assert book.metadata["location_end"] == 30
    assert book.metadata["source_files"] == ["My Clippings.txt"]


def test_kindle_clippings_author_aggregate_groups_books_and_edges_respect_filters(tmp_path):
    clippings = tmp_path / "My Clippings.txt"
    clippings.write_text(
        """First Book (Ada Author)
- Your Highlight at location 10 | Added on Monday, January 1, 2024 1:00:00 PM

Highlight.
==========
Second Book (Ada Author)
- Your Note at location 20 | Added on Tuesday, January 2, 2024 2:00:00 PM

Note.
==========
Second Book (Ada Author)
- Your Bookmark at location 25 | Added on Wednesday, January 3, 2024 3:00:00 PM

==========
Other Book (Other Author)
- Your Highlight at location 30 | Added on Thursday, January 4, 2024 4:00:00 PM

Other.
==========
""",
        encoding="utf-8",
    )

    author_only = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["author"])
    combined = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["author", "book"])

    ada_author = next(unit for unit in author_only.units if unit.metadata["author"] == "Ada Author")
    assert {unit.source_entity_type for unit in author_only.units} == {"author"}
    assert ada_author.metadata["book_count"] == 2
    assert ada_author.metadata["clipping_count"] == 3
    assert ada_author.metadata["highlight_count"] == 1
    assert ada_author.metadata["note_count"] == 1
    assert ada_author.metadata["bookmark_count"] == 1
    assert ada_author.metadata["first_clipped_at"] == "2024-01-01T13:00:00+00:00"
    assert ada_author.metadata["last_clipped_at"] == "2024-01-03T15:00:00+00:00"
    assert ada_author.metadata["source_files"] == ["My Clippings.txt"]
    assert len(ada_author.metadata["book_source_ids"]) == 2
    assert author_only.edges == []

    ada_combined = next(unit for unit in combined.units if unit.source_entity_type == "author" and unit.title == "Ada Author")
    ada_edges = [
        edge
        for edge in combined.edges
        if edge.from_unit_id == ada_combined.source_id and edge.metadata["relation_type"] == "author_contains_book"
    ]
    assert len(ada_edges) == 2
    assert {edge.to_unit_id for edge in ada_edges} == set(ada_combined.metadata["book_source_ids"])


def test_kindle_clippings_ingests_notebook_html_exports(tmp_path):
    notebook = tmp_path / "notebook.html"
    notebook.write_text(
        """<html><body>
<h1>HTML Book (Ada Author)</h1>
<div>Highlight (page 12 | location 120-121)</div>
<div>HTML highlighted text.</div>
<div>Added on January 2, 2024 03:04:05 PM</div>
<div>Note (location 122)</div>
<div>HTML note text.</div>
<div>Added on 2024-01-03 04:05:06</div>
</body></html>""",
        encoding="utf-8",
    )

    result = KindleClippingsAdapter(path=str(notebook)).ingest()

    clippings = [unit for unit in result.units if unit.source_entity_type == "clipping"]
    assert len(clippings) == 2
    highlight = next(unit for unit in clippings if unit.metadata["clipping_type"] == "highlight")
    note = next(unit for unit in clippings if unit.metadata["clipping_type"] == "note")
    assert highlight.content == "HTML highlighted text."
    assert highlight.metadata["book_title"] == "HTML Book"
    assert highlight.metadata["author"] == "Ada Author"
    assert highlight.metadata["page"] == "12"
    assert highlight.metadata["location"] == "120-121"
    assert highlight.metadata["source_file"] == "notebook.html"
    assert highlight.created_at == datetime(2024, 1, 2, 15, 4, 5, tzinfo=timezone.utc)
    assert note.content == "HTML note text."
    assert note.metadata["location"] == "122"
    book = next(unit for unit in result.units if unit.source_entity_type == "book")
    assert book.metadata["highlight_count"] == 1
    assert book.metadata["note_count"] == 1
    assert len(result.edges) == 4
    assert [edge.metadata["relation_type"] for edge in result.edges if edge.metadata["relation_type"] == "note_references_highlight"] == [
        "note_references_highlight"
    ]


def test_kindle_clippings_directory_discovers_txt_html_and_htm(tmp_path):
    (tmp_path / "My Clippings.txt").write_text(
        """Text Book (Author)
- Your Highlight at location 1 | Added on Monday, January 1, 2024 1:00:00 PM

Text highlight.
==========
""",
        encoding="utf-8",
    )
    (tmp_path / "notebook.htm").write_text(
        """<h1>HTML Book</h1><div>Highlight (location 2)</div><div>HTML highlight.</div>""",
        encoding="utf-8",
    )

    result = KindleClippingsAdapter(path=str(tmp_path)).ingest(entity_types=["clipping"])

    assert {unit.content for unit in result.units} == {"Text highlight.", "HTML highlight."}


def test_kindle_clippings_adapter_is_registered():
    assert "kindle_clippings" in list_adapters()
    adapter = get_adapter("kindle_clippings", path="/tmp/My Clippings.txt")
    assert isinstance(adapter, KindleClippingsAdapter)
    assert adapter.name == "kindle_clippings"


def test_kindle_clippings_links_notes_to_exact_and_nearby_highlights(tmp_path):
    clippings = tmp_path / "My Clippings.txt"
    clippings.write_text(
        """Shared Book (Author)
- Your Highlight at location 100-102 | Added on Monday, January 1, 2024 1:00:00 PM

Highlight one.
==========
Shared Book (Author)
- Your Note at location 100-102 | Added on Monday, January 1, 2024 1:01:00 PM

Exact note.
==========
Shared Book (Author)
- Your Highlight at location 200-202 | Added on Monday, January 1, 2024 2:00:00 PM

Highlight two.
==========
Shared Book (Author)
- Your Note at location 204 | Added on Monday, January 1, 2024 2:01:00 PM

Nearby note.
==========
Shared Book (Author)
- Your Note at location 400 | Added on Monday, January 1, 2024 3:01:00 PM

Unmatched note.
==========
""",
        encoding="utf-8",
    )

    result = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["clipping"])

    annotation_edges = [edge for edge in result.edges if edge.metadata["relation_type"] == "note_references_highlight"]
    assert len(annotation_edges) == 2
    assert {edge.metadata["match_strategy"] for edge in annotation_edges} == {"exact", "nearby"}
    assert {edge.metadata["book_title"] for edge in annotation_edges} == {"Shared Book"}
    assert len([unit for unit in result.units if unit.metadata["clipping_type"] == "note"]) == 3


def test_kindle_clippings_author_aggregates_and_edges(tmp_path):
    clippings = tmp_path / "My Clippings.txt"
    clippings.write_text(
        """First Book (Ada Author)
- Your Highlight at location 10 | Added on Monday, January 1, 2024 1:00:00 PM

Highlight one.
==========
Second Book (Ada Author)
- Your Note at location 20 | Added on Tuesday, January 2, 2024 2:00:00 PM

Note one.
==========
""",
        encoding="utf-8",
    )

    result = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["author", "book"])
    author = next(unit for unit in result.units if unit.source_entity_type == "author")
    books = [unit for unit in result.units if unit.source_entity_type == "book"]

    assert author.metadata["book_count"] == 2
    assert author.metadata["clipping_count"] == 2
    assert author.metadata["highlight_count"] == 1
    assert author.metadata["note_count"] == 1
    assert author.metadata["book_source_ids"] == [book.source_id for book in books]
    assert len(result.edges) == 2
    assert all(edge.metadata["relation_type"] == "author_contains_book" for edge in result.edges)


def test_kindle_clippings_emits_clipping_type_aggregates_and_edges(tmp_path):
    clippings = tmp_path / "My Clippings.txt"
    clippings.write_text(
        """First Book (Ada Author)
- Your Highlight at location 10-12 | Added on Monday, January 1, 2024 1:00:00 PM

Highlight one.
==========
Second Book (Grace Author)
- Your Highlight at location 30 | Added on Tuesday, January 2, 2024 2:00:00 PM

Highlight two.
==========
First Book (Ada Author)
- Your Note at location 20 | Added on Wednesday, January 3, 2024 3:00:00 PM

Note one.
==========
First Book (Ada Author)
- Your Bookmark at location 40 | Added on Thursday, January 4, 2024 4:00:00 PM

==========
""",
        encoding="utf-8",
    )

    result = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["clipping", "clipping_type"])

    assert "clipping_type" in KindleClippingsAdapter(path=str(clippings)).entity_types
    types = {unit.metadata["clipping_type"]: unit for unit in result.units if unit.source_entity_type == "clipping_type"}
    assert set(types) == {"highlight", "note", "bookmark"}
    highlight = types["highlight"]
    assert highlight.metadata["clipping_count"] == 2
    assert highlight.metadata["book_count"] == 2
    assert highlight.metadata["authors"] == ["Ada Author", "Grace Author"]
    assert highlight.metadata["source_files"] == ["My Clippings.txt"]
    assert highlight.metadata["first_added_at"] == "2024-01-01T13:00:00+00:00"
    assert highlight.metadata["latest_added_at"] == "2024-01-02T14:00:00+00:00"
    assert highlight.metadata["location_start"] == 10
    assert highlight.metadata["location_end"] == 30
    type_edges = [edge for edge in result.edges if edge.metadata.get("relation_type") == "clipping_type_contains_clipping"]
    assert len(type_edges) == 4
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in type_edges)

    type_only = KindleClippingsAdapter(path=str(clippings)).ingest(entity_types=["clipping_type"])
    assert {unit.source_entity_type for unit in type_only.units} == {"clipping_type"}
    assert type_only.edges == []
