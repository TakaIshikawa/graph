from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.letterboxd_ratings_csv import LetterboxdRatingsCsvAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_letterboxd_ratings_csv_ingests_representative_rows(tmp_path):
    export = tmp_path / "ratings.csv"
    export.write_text(
        "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Review URL,Tags\n"
        "2026-05-01,In the Mood for Love,2000,https://letterboxd.com/film/in-the-mood-for-love/,5,No,https://letterboxd.com/me/film/in-the-mood-for-love/,romance;favorites\n"
        "2026-05-02,Heat,1995,https://letterboxd.com/film/heat-1995/,4.5,Yes,,crime|rewatch\n",
        encoding="utf-8",
    )

    result = LetterboxdRatingsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    unit = result.units[0]
    assert unit.source_project == SourceProject.LETTERBOXD
    assert unit.source_entity_type == "film_rating"
    assert unit.content_type == ContentType.METADATA
    assert unit.title == "In the Mood for Love (2000)"
    assert unit.metadata["title"] == "In the Mood for Love"
    assert unit.metadata["year"] == "2000"
    assert unit.metadata["rating"] == "5"
    assert unit.metadata["watched_date"] == "2026-05-01"
    assert unit.metadata["rewatch"] is False
    assert unit.metadata["review_url"] == "https://letterboxd.com/me/film/in-the-mood-for-love/"
    assert unit.metadata["letterboxd_uri"] == "https://letterboxd.com/film/in-the-mood-for-love/"
    assert unit.metadata["tags"] == ["romance", "favorites"]
    assert unit.metadata["source_file"] == "ratings.csv"
    assert unit.metadata["source_row"]["Rating"] == "5"
    assert unit.created_at == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert {"letterboxd", "film_rating", "romance", "favorites"}.issubset(set(unit.tags))
    assert "Review URL: https://letterboxd.com/me/film/in-the-mood-for-love/" in unit.content
    assert result.units[1].metadata["rewatch"] is True


def test_letterboxd_ratings_csv_handles_sparse_rows_and_stable_ids(tmp_path):
    export = tmp_path / "ratings.csv"
    export.write_text(
        "Name,Year,Rating,Date,Letterboxd URI,Review URL,Rewatch,Tags\n"
        ",,,,https://letterboxd.com/film/unknown/,,,,\n"
        "Sparse Film,,3,,,,,\n"
        ",,,,,,,\n",
        encoding="utf-8",
    )

    first = LetterboxdRatingsCsvAdapter(path=str(export)).ingest()
    second = LetterboxdRatingsCsvAdapter(path=str(export)).ingest()

    assert [unit.title for unit in first.units] == ["Untitled Letterboxd rating", "Sparse Film"]
    assert first.units[0].metadata["letterboxd_uri"] == "https://letterboxd.com/film/unknown/"
    assert first.units[1].metadata["rating"] == "3"
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]


def test_letterboxd_ratings_csv_directory_since_and_entity_filter(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("Date,Name,Year,Rating\n2026-05-01,Old,2020,2\n", encoding="utf-8")
    new = tmp_path / "new.csv"
    new.write_text("Date,Name,Year,Rating\n2026-05-03,New,2021,4\n", encoding="utf-8")
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff\xfe\x00")
    ignored = tmp_path / "notes.txt"
    ignored.write_text("Date,Name,Year,Rating\n2026-05-04,Ignored,2022,5\n", encoding="utf-8")
    since = SyncState(source_project="letterboxd", source_entity_type="film_rating", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = LetterboxdRatingsCsvAdapter(path=str(tmp_path)).ingest(since=since)

    assert LetterboxdRatingsCsvAdapter().entity_types == ["film_rating"]
    assert [unit.metadata["title"] for unit in result.units] == ["New"]
    assert LetterboxdRatingsCsvAdapter(path=str(tmp_path)).ingest(entity_types=["diary_entry"]).units == []
