from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.google_play_books_notes_csv import GooglePlayBooksNotesCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_google_play_books_notes_csv_ingests_highlight_and_note(tmp_path):
    export = tmp_path / "notes.csv"
    _write_csv(
        export,
        [
            {
                "Book Title": "Example Book",
                "Author": "A. Author",
                "Highlight": "Highlighted passage",
                "Note": "My note",
                "Color": "Yellow",
                "Page": "42",
                "Created At": "2026-05-01T10:00:00Z",
            }
        ],
    )

    result = GooglePlayBooksNotesCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GOOGLE_PLAY_BOOKS_NOTES_CSV
    assert unit.source_entity_type == "book_note"
    assert unit.title == "Google Play Books note: Example Book"
    assert unit.metadata["book_title"] == "Example Book"
    assert unit.metadata["highlight"] == "Highlighted passage"
    assert unit.metadata["note"] == "My note"
    assert unit.metadata["color"] == "Yellow"
    assert unit.metadata["location"] == "42"
    assert unit.metadata["created_at"] == "2026-05-01T10:00:00+00:00"
    assert unit.updated_at == datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    assert "Highlight:\nHighlighted passage" in unit.content


def test_google_play_books_notes_csv_handles_sparse_rows_filters_and_registry(tmp_path):
    _write_csv(
        tmp_path / "notes.csv",
        [
            {"Title": "Old", "Highlighted Text": "Old highlight", "Date": "2026-04-30"},
            {"Title": "New", "Notes": "Note only", "Location": "Loc 1", "Date": "2026-05-03"},
            {"Title": "No timestamp", "Highlight": "Timeless"},
            {"Title": "No useful content", "Page": "2"},
        ],
    )
    since = SyncState(
        source_project="google_play_books_notes_csv",
        source_entity_type="book_note",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = GooglePlayBooksNotesCsvAdapter(path=str(tmp_path)).ingest(since=since)
    skipped = GooglePlayBooksNotesCsvAdapter(path=str(tmp_path)).ingest(entity_types=["book"])

    assert [unit.metadata["book_title"] for unit in result.units] == ["New", "No timestamp"]
    assert result.units[0].metadata["note"] == "Note only"
    assert result.units[1].metadata["highlight"] == "Timeless"
    assert skipped.units == []
    assert get_adapter("google_play_books_notes_csv", path=str(tmp_path)).name == "google_play_books_notes_csv"


def test_google_play_books_notes_csv_source_id_is_deterministic(tmp_path):
    export = tmp_path / "notes.csv"
    _write_csv(export, [{"Title": "Book", "Highlight": "Same", "Location": "p1"}])

    first = GooglePlayBooksNotesCsvAdapter(path=str(export)).ingest().units[0]
    second = GooglePlayBooksNotesCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
