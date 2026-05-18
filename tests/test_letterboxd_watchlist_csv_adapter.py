from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.letterboxd_watchlist_csv import LetterboxdWatchlistCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_letterboxd_watchlist_csv_ingests_watchlist_metadata(tmp_path):
    export = tmp_path / "watchlist.csv"
    export.write_text(
        "Date,Name,Year,Letterboxd URI\n"
        "2026-05-01,Heat,1995,https://letterboxd.com/film/heat-1995/\n"
        "2026-05-02,Heat,1986,https://letterboxd.com/film/heat-1986/\n",
        encoding="utf-8",
    )

    result = LetterboxdWatchlistCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    first = result.units[0]
    assert first.source_project == "letterboxd_watchlist_csv"
    assert first.source_entity_type == "watchlist_item"
    assert first.content_type == ContentType.METADATA
    assert first.title == "Heat (1995)"
    assert first.metadata["title"] == "Heat"
    assert first.metadata["year"] == "1995"
    assert first.metadata["added_date"] == "2026-05-01T00:00:00+00:00"
    assert first.metadata["url"] == "https://letterboxd.com/film/heat-1995/"
    assert first.metadata["letterboxd_uri"] == "https://letterboxd.com/film/heat-1995/"
    assert first.metadata["watchlist"] is True
    assert first.tags == ["letterboxd", "watchlist"]
    assert "Film: Heat (1995)" in first.content
    assert "URL: https://letterboxd.com/film/heat-1995/" in first.content
    assert result.units[0].source_id != result.units[1].source_id


def test_letterboxd_watchlist_csv_keeps_same_title_different_years_distinct_without_urls(tmp_path):
    export = tmp_path / "watchlist.csv"
    export.write_text(
        "Date,Name,Year\n"
        "2026-05-01,The Thing,1951\n"
        "2026-05-01,The Thing,1982\n",
        encoding="utf-8",
    )

    units = LetterboxdWatchlistCsvAdapter(path=str(export)).ingest().units

    assert [unit.title for unit in units] == ["The Thing (1951)", "The Thing (1982)"]
    assert len({unit.source_id for unit in units}) == 2


def test_letterboxd_watchlist_csv_skips_blank_rows_and_filters_since(tmp_path):
    export = tmp_path / "watchlist.csv"
    export.write_text(
        "Date,Name,Year,Letterboxd URI\n"
        ",,,\n"
        "2026-05-01,Old Film,2001,https://letterboxd.com/film/old-film/\n"
        "2026-05-03,New Film,2003,https://letterboxd.com/film/new-film/\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="letterboxd_watchlist_csv",
        source_entity_type="watchlist_item",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    filtered = LetterboxdWatchlistCsvAdapter(path=str(export)).ingest(since=since).units

    assert [unit.title for unit in filtered] == ["New Film (2003)"]
    assert LetterboxdWatchlistCsvAdapter(path=str(export)).ingest(entity_types=["film"]).units == []
