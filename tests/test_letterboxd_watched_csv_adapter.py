from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.letterboxd_watched_csv import LetterboxdWatchedCsvAdapter


def test_letterboxd_watched_csv_ingests_representative_row(tmp_path):
    export = tmp_path / "watched.csv"
    export.write_text(
        "Name,Year,Letterboxd URI,Watched Date,Rating,Rewatch,Tags,Review\n"
        'Aftersun,2022,https://boxd.it/film/aftersun/,2025-01-02,4.5,Yes,"drama, favorite",Quietly devastating.\n',
        encoding="utf-8",
    )

    unit = LetterboxdWatchedCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "letterboxd_watched_csv"
    assert unit.source_entity_type == "film"
    assert unit.title == "Aftersun (2022)"
    assert "Rating: 4.5" in unit.content
    assert "Review: Quietly devastating." in unit.content
    assert unit.metadata["year"] == "2022"
    assert unit.metadata["letterboxd_uri"] == "https://boxd.it/film/aftersun/"
    assert unit.metadata["rating"] == 4.5
    assert unit.metadata["rewatch"] is True
    assert unit.metadata["tags"] == ["drama", "favorite"]
    assert unit.tags == ["drama", "favorite"]
    assert unit.created_at == datetime(2025, 1, 2, tzinfo=timezone.utc)


def test_letterboxd_watched_csv_handles_missing_optional_fields(tmp_path):
    export = tmp_path / "watched.csv"
    export.write_text("Name,Year\nFilm Only,1999\n", encoding="utf-8")

    unit = LetterboxdWatchedCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Film Only (1999)"
    assert "rating" not in unit.metadata
    assert "review" not in unit.metadata
    assert unit.tags == []
