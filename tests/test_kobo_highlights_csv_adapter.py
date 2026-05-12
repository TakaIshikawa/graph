from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.kobo_highlights_csv import KoboHighlightsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_kobo_highlights_csv_ingests_highlights_and_notes(tmp_path):
    path = tmp_path / "kobo.csv"
    _write_csv(
        path,
        [
            {
                "Book Title": "The Left Hand of Darkness",
                "Author": "Ursula K. Le Guin",
                "ISBN": "9780441478125",
                "Annotation": "The king was pregnant.",
                "Note": "Important opening.",
                "Color": "Yellow",
                "Chapter": "1",
                "Page": "12",
                "Date Created": "2026-05-01T10:00:00Z",
                "Date Modified": "2026-05-02T10:00:00Z",
                "Book URL": "https://example.test/book",
            }
        ],
    )

    result = KoboHighlightsCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.KOBO_HIGHLIGHTS_CSV
    assert unit.source_entity_type == "highlight"
    assert unit.metadata["book_title"] == "The Left Hand of Darkness"
    assert unit.metadata["author"] == "Ursula K. Le Guin"
    assert unit.metadata["isbn"] == "9780441478125"
    assert unit.metadata["highlight"] == "The king was pregnant."
    assert unit.metadata["note"] == "Important opening."
    assert unit.metadata["chapter"] == "1"
    assert unit.metadata["location"] == "12"
    assert unit.metadata["color"] == "Yellow"
    assert unit.metadata["date_created"] == "2026-05-01T10:00:00+00:00"
    assert unit.metadata["date_modified"] == "2026-05-02T10:00:00+00:00"
    assert unit.metadata["book_url"] == "https://example.test/book"
    assert unit.metadata["row"]["Note"] == "Important opening."
    assert unit.created_at == datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 2, 10, tzinfo=timezone.utc)


def test_kobo_highlights_csv_tolerates_missing_optional_columns(tmp_path):
    path = tmp_path / "kobo.csv"
    _write_csv(path, [{"Book Title": "Sparse Book", "Highlighted Text": "Sparse highlight"}])

    result = KoboHighlightsCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Kobo highlight: Sparse Book"
    assert unit.metadata["highlight"] == "Sparse highlight"
    assert unit.metadata["note"] == ""
    assert unit.metadata["color"] == ""
    assert unit.metadata["chapter"] == ""
    assert unit.metadata["location"] == ""


def test_kobo_highlights_csv_directory_since_filter_and_blank_skip(tmp_path):
    _write_csv(tmp_path / "old.csv", [{"Book Title": "Old", "Annotation": "old", "Date Created": "2026-04-01"}])
    _write_csv(tmp_path / "new.csv", [{"Book Title": "New", "Note": "new note", "Date Created": "2026-05-01"}])
    _write_csv(tmp_path / "blank.csv", [{"Book Title": "Blank", "Annotation": "", "Note": ""}])
    since = SyncState(source_project="kobo_highlights_csv", source_entity_type="highlight", last_sync_at=datetime(2026, 4, 15, tzinfo=timezone.utc))

    result = KoboHighlightsCsvAdapter(path=str(tmp_path)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["Kobo note: New"]
    assert result.units[0].source_entity_type == "note"
    assert get_adapter("kobo_highlights_csv", path=str(tmp_path)).name == "kobo_highlights_csv"


def test_kobo_highlights_csv_filters_entity_types_and_stable_ids(tmp_path):
    path = tmp_path / "kobo.csv"
    _write_csv(path, [{"Book Title": "Book", "Annotation": "highlight", "Date Created": "2026-05-01"}])

    first = KoboHighlightsCsvAdapter(path=str(path)).ingest().units[0]
    second = KoboHighlightsCsvAdapter(path=str(path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert KoboHighlightsCsvAdapter(path=str(path)).ingest(entity_types=["note"]).units == []
