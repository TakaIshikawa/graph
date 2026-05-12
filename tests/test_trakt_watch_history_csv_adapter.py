from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.trakt_watch_history_csv import TraktWatchHistoryCsvAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_trakt_watch_history_csv_ingests_movie_rows(tmp_path):
    export = tmp_path / "history.csv"
    _write_csv(
        export,
        [
            {
                "watched_at": "2025-01-02T03:04:05Z",
                "title": "Arrival",
                "year": "2016",
                "type": "movie",
                "imdb_id": "tt2543164",
                "tmdb_id": "329865",
                "trakt_id": "12345",
                "url": "https://trakt.tv/movies/arrival-2016",
                "rating": "10",
            }
        ],
    )

    result = TraktWatchHistoryCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.TRAKT_WATCH_HISTORY_CSV
    assert unit.title == "Arrival (2016)"
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.metadata["watched_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["imdb_id"] == "tt2543164"
    assert unit.metadata["tmdb_id"] == "329865"
    assert unit.metadata["trakt_id"] == "12345"
    assert unit.metadata["urls"] == {"url": "https://trakt.tv/movies/arrival-2016"}
    assert unit.metadata["row"]["rating"] == "10"
    assert unit.metadata["watch_sequence"] == 1
    assert unit.metadata["is_rewatch"] is False
    assert "previous_watch_at" not in unit.metadata


def test_trakt_watch_history_csv_ingests_episode_rows_and_filters(tmp_path):
    export = tmp_path / "history.csv"
    _write_csv(
        export,
        [
            {
                "watched_at": "2024-12-31T23:00:00Z",
                "title": "Old Episode",
                "type": "episode",
                "season": "1",
                "episode": "1",
            },
            {
                "watched_at": "2025-02-03T04:05:06+00:00",
                "title": "The One with the Adapter",
                "year": "2025",
                "type": "episode",
                "season": "2",
                "episode": "7",
                "trakt_url": "https://trakt.tv/shows/example/seasons/2/episodes/7",
            },
        ],
    )
    since = SyncState(
        source_project="trakt_watch_history_csv",
        source_entity_type="watch",
        last_sync_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    result = TraktWatchHistoryCsvAdapter(path=str(export)).ingest(since=since)

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "The One with the Adapter (2025) S02E07"
    assert unit.metadata["season"] == 2
    assert unit.metadata["episode"] == 7
    assert "Season: 2" in unit.content
    assert unit.metadata["urls"]["trakt_url"].endswith("/episodes/7")
    assert TraktWatchHistoryCsvAdapter(path=str(export)).ingest(entity_types=["movie"]).units == []


def test_trakt_watch_history_csv_directory_and_registry(tmp_path):
    _write_csv(tmp_path / "one.csv", [{"watched_at": "2025-01-01T00:00:00Z", "title": "One", "type": "movie"}])
    _write_csv(tmp_path / "two.csv", [{"watched_at": "2025-01-02T00:00:00Z", "title": "Two", "type": "movie"}])

    result = TraktWatchHistoryCsvAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert get_adapter("trakt_watch_history_csv", path=str(tmp_path)).name == "trakt_watch_history_csv"


def test_trakt_watch_history_csv_adds_rewatch_sequence_in_chronological_order(tmp_path):
    export = tmp_path / "history.csv"
    _write_csv(
        export,
        [
            {
                "watched_at": "2025-03-10T12:00:00Z",
                "title": "Arrival",
                "year": "2016",
                "type": "movie",
                "trakt_id": "movie-1",
            },
            {
                "watched_at": "2025-01-10T12:00:00Z",
                "title": "Arrival",
                "year": "2016",
                "type": "movie",
                "trakt_id": "movie-1",
            },
            {
                "watched_at": "2025-02-10T12:00:00Z",
                "title": "Arrival",
                "year": "2016",
                "type": "movie",
                "trakt_id": "movie-1",
            },
        ],
    )

    units = TraktWatchHistoryCsvAdapter(path=str(export)).ingest().units

    assert [unit.metadata["watch_sequence"] for unit in units] == [1, 2, 3]
    assert [unit.metadata["is_rewatch"] for unit in units] == [False, True, True]
    assert "previous_watch_at" not in units[0].metadata
    assert units[1].metadata["previous_watch_at"] == "2025-01-10T12:00:00+00:00"
    assert units[2].metadata["previous_watch_at"] == "2025-02-10T12:00:00+00:00"


def test_trakt_watch_history_csv_rewatch_identity_distinguishes_episodes(tmp_path):
    export = tmp_path / "history.csv"
    _write_csv(
        export,
        [
            {
                "watched_at": "2025-01-01T00:00:00Z",
                "title": "Example Show",
                "year": "2025",
                "type": "episode",
                "season": "1",
                "episode": "1",
            },
            {
                "watched_at": "2025-01-02T00:00:00Z",
                "title": "Example Show",
                "year": "2025",
                "type": "episode",
                "season": "1",
                "episode": "2",
            },
            {
                "watched_at": "2025-01-03T00:00:00Z",
                "title": "Example Show",
                "year": "2025",
                "type": "episode",
                "season": "1",
                "episode": "1",
            },
        ],
    )

    units = TraktWatchHistoryCsvAdapter(path=str(export)).ingest().units

    assert [unit.title for unit in units] == [
        "Example Show (2025) S01E01",
        "Example Show (2025) S01E02",
        "Example Show (2025) S01E01",
    ]
    assert [unit.metadata["watch_sequence"] for unit in units] == [1, 1, 2]
    assert [unit.metadata["is_rewatch"] for unit in units] == [False, False, True]
    assert units[2].metadata["previous_watch_at"] == "2025-01-01T00:00:00+00:00"
