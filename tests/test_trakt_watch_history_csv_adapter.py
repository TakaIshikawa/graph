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

    result = TraktWatchHistoryCsvAdapter(path=str(export)).ingest(entity_types=["watch"])

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

    result = TraktWatchHistoryCsvAdapter(path=str(export)).ingest(since=since, entity_types=["watch"])

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

    result = TraktWatchHistoryCsvAdapter(path=str(tmp_path)).ingest(entity_types=["watch"])

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

    units = TraktWatchHistoryCsvAdapter(path=str(export)).ingest(entity_types=["watch"]).units

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

    units = TraktWatchHistoryCsvAdapter(path=str(export)).ingest(entity_types=["watch"]).units

    assert [unit.title for unit in units] == [
        "Example Show (2025) S01E01",
        "Example Show (2025) S01E02",
        "Example Show (2025) S01E01",
    ]
    assert [unit.metadata["watch_sequence"] for unit in units] == [1, 1, 2]
    assert [unit.metadata["is_rewatch"] for unit in units] == [False, False, True]
    assert units[2].metadata["previous_watch_at"] == "2025-01-01T00:00:00+00:00"


def test_trakt_watch_history_csv_media_aggregates_repeated_watches(tmp_path):
    export = tmp_path / "history.csv"
    _write_csv(
        export,
        [
            {
                "watched_at": "2025-01-01T00:00:00Z",
                "title": "Arrival",
                "year": "2016",
                "type": "movie",
                "trakt_id": "movie-1",
                "url": "https://trakt.tv/movies/arrival-2016",
            },
            {
                "watched_at": "2025-02-01T00:00:00Z",
                "title": "Arrival",
                "year": "2016",
                "type": "movie",
                "trakt_id": "movie-1",
                "url": "https://trakt.tv/movies/arrival-2016",
            },
        ],
    )

    result = TraktWatchHistoryCsvAdapter(path=str(export)).ingest(entity_types=["media", "watch"])

    media = [unit for unit in result.units if unit.source_entity_type == "media"]
    watches = [unit for unit in result.units if unit.source_entity_type == "watch"]
    assert len(media) == 1
    assert len(watches) == 2
    unit = media[0]
    assert unit.source_id.startswith("trakt_watch_history_csv:media:")
    assert unit.metadata["watch_count"] == 2
    assert unit.metadata["rewatch_count"] == 1
    assert unit.metadata["first_watched_at"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["last_watched_at"] == "2025-02-01T00:00:00+00:00"
    assert unit.metadata["identifiers"] == {"trakt_id": "movie-1"}
    assert len(result.edges) == 2
    assert all(edge.from_unit_id == unit.source_id for edge in result.edges)


def test_trakt_watch_history_csv_ingests_ratings_and_merges_media(tmp_path):
    export = tmp_path / "ratings.csv"
    _write_csv(
        export,
        [
            {
                "rated_at": "2025-01-03T00:00:00Z",
                "title": "Arrival",
                "year": "2016",
                "type": "movie",
                "trakt_id": "movie-1",
                "rating": "9",
            },
            {
                "watched_at": "2025-01-01T00:00:00Z",
                "title": "Arrival",
                "year": "2016",
                "type": "movie",
                "trakt_id": "movie-1",
            },
        ],
    )

    result = TraktWatchHistoryCsvAdapter(path=str(export)).ingest(entity_types=["media", "watch", "rating"])

    ratings = [unit for unit in result.units if unit.source_entity_type == "rating"]
    watches = [unit for unit in result.units if unit.source_entity_type == "watch"]
    media = [unit for unit in result.units if unit.source_entity_type == "media"]
    assert len(ratings) == 1
    assert len(watches) == 1
    assert len(media) == 1
    assert ratings[0].source_id.startswith("trakt_watch_history_csv:rating:")
    assert ratings[0].metadata["rating"] == 9
    assert ratings[0].metadata["rated_at"] == "2025-01-03T00:00:00+00:00"
    assert media[0].metadata["watch_count"] == 1
    assert media[0].metadata["rating_count"] == 1
    assert media[0].metadata["ratings"] == [9]
    assert media[0].metadata["last_rated_at"] == "2025-01-03T00:00:00+00:00"
    assert {edge.metadata["relation_type"] for edge in result.edges} == {
        "media_contains_watch",
        "media_contains_rating",
    }


def test_trakt_watch_history_csv_rating_edges_follow_entity_filters(tmp_path):
    export = tmp_path / "ratings.csv"
    _write_csv(
        export,
        [{"rated_at": "2025-01-03T00:00:00Z", "title": "Arrival", "type": "movie", "rating": "8"}],
    )

    rating_only = TraktWatchHistoryCsvAdapter(path=str(export)).ingest(entity_types=["rating"])
    media_and_rating = TraktWatchHistoryCsvAdapter(path=str(export)).ingest(entity_types=["media", "rating"])

    assert [unit.source_entity_type for unit in rating_only.units] == ["rating"]
    assert rating_only.edges == []
    assert {unit.source_entity_type for unit in media_and_rating.units} == {"media", "rating"}
    assert [edge.metadata["relation_type"] for edge in media_and_rating.edges] == ["media_contains_rating"]
