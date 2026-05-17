from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.letterboxd_diary_csv import LetterboxdDiaryCsvAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_letterboxd_diary_csv_imports_entries(tmp_path):
    path = tmp_path / "diary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Date", "Name", "Year", "Letterboxd URI", "Rating", "Rewatch", "Tags", "Review"])
        writer.writeheader()
        writer.writerow(
            {
                "Date": "2025-01-15",
                "Name": "Inception",
                "Year": "2010",
                "Letterboxd URI": "https://letterboxd.com/film/inception/",
                "Rating": "4.5",
                "Rewatch": "No",
                "Tags": "sci-fi, thriller",
                "Review": "Mind-bending.",
            }
        )

    unit = LetterboxdDiaryCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == SourceProject.LETTERBOXD
    assert unit.source_entity_type == "diary_entry"
    assert unit.title == "Inception (2010)"
    assert unit.content == "Film: Inception (2010)\nWatched date: 2025-01-15\nRating: 4.5\nTags: sci-fi, thriller\n\nReview:\nMind-bending."
    assert unit.metadata["title"] == "Inception"
    assert unit.metadata["year"] == "2010"
    assert unit.metadata["watched_date"] == "2025-01-15"
    assert unit.metadata["rating"] == "4.5"
    assert unit.metadata["rewatch"] is False
    assert unit.metadata["tags"] == ["sci-fi", "thriller"]
    assert unit.metadata["source_file"] == str(path)
    assert unit.metadata["row_index"] == 0
    assert unit.metadata["row"]["Review"] == "Mind-bending."
    assert unit.created_at == datetime(2025, 1, 15, tzinfo=timezone.utc)


def test_letterboxd_diary_csv_directory_since_and_stable_ids(tmp_path):
    old = tmp_path / "old.csv"
    new = tmp_path / "new.csv"
    for path, title, date in ((old, "Old", "2025-01-01"), (new, "New", "2025-01-03")):
        path.write_text(f"Date,Name,Year\n{date},{title},2020\n", encoding="utf-8")

    since = SyncState(
        source_project="letterboxd",
        source_entity_type="diary_entry",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    filtered = LetterboxdDiaryCsvAdapter(path=str(tmp_path)).ingest(since=since)
    first = LetterboxdDiaryCsvAdapter(path=str(new)).ingest().units[0]
    second = LetterboxdDiaryCsvAdapter(path=str(new)).ingest().units[0]

    assert [unit.metadata["title"] for unit in filtered.units] == ["New"]
    assert first.source_id == second.source_id


def test_letterboxd_diary_csv_skips_titleless_and_respects_entity_filter(tmp_path):
    path = tmp_path / "diary.csv"
    path.write_text("Date,Name,Year\n2025-01-01,,2020\n2025-01-02,Kept,2021\n", encoding="utf-8")

    result = LetterboxdDiaryCsvAdapter(path=str(path)).ingest()
    wrong_entity = LetterboxdDiaryCsvAdapter(path=str(path)).ingest(entity_types=["film"])

    assert [unit.metadata["title"] for unit in result.units] == ["Kept"]
    assert wrong_entity.units == []
