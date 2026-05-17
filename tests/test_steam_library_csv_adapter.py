from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.steam_library_csv import SteamLibraryCsvAdapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_steam_library_csv_ingests_complete_rows_and_normalizes_playtime(tmp_path):
    export = tmp_path / "steam.csv"
    _write_csv(
        export,
        [
            {
                "App ID": "620",
                "Name": "Portal 2",
                "Hours Played": "12.5",
                "Last Played": "2025-01-02T03:04:05Z",
                "Store URL": "https://store.steampowered.com/app/620/Portal_2/",
                "Tags": "Puzzle; Co-op",
            }
        ],
    )

    result = SteamLibraryCsvAdapter(path=str(export)).ingest(entity_types=["game"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.STEAM_LIBRARY_CSV
    assert unit.source_id == "steam_library_csv:620"
    assert unit.title == "Portal 2"
    assert unit.metadata["app_id"] == "620"
    assert unit.metadata["playtime_minutes"] == 750
    assert unit.metadata["playtime_bucket"] == "deep_play"
    assert unit.metadata["last_played"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["store_url"].endswith("/Portal_2/")
    assert unit.metadata["platform"] == "steam"
    assert unit.metadata["row"]["Hours Played"] == "12.5"
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert "puzzle" in unit.tags


def test_steam_library_csv_supports_aliases_sparse_rows_and_registry(tmp_path):
    export = tmp_path / "library.csv"
    _write_csv(
        export,
        [
            {"appid": "400", "game": "Portal", "playtime_forever": "123", "categories": "Single-player"},
            {"title": "No App Id Game"},
        ],
    )

    result = SteamLibraryCsvAdapter(path=str(export)).ingest(entity_types=["game"])

    assert len(result.units) == 2
    first = result.units[0]
    assert first.source_id == "steam_library_csv:400"
    assert first.metadata["playtime_minutes"] == 123
    assert first.metadata["playtime_bucket"] == "played"
    assert first.metadata["store_url"] == "https://store.steampowered.com/app/400/"
    assert "last_played" not in first.metadata
    assert result.units[1].title == "No App Id Game"
    assert get_adapter("steam_library_csv", path=str(export)).name == "steam_library_csv"


def test_steam_library_csv_playtime_buckets_and_invalid_values(tmp_path):
    export = tmp_path / "steam.csv"
    _write_csv(
        export,
        [
            {"App ID": "1", "Name": "Zero", "Minutes Played": "0"},
            {"App ID": "2", "Name": "Short", "Minutes Played": "59"},
            {"App ID": "3", "Name": "Normal", "Minutes Played": "60"},
            {"App ID": "4", "Name": "High", "Hours Played": "10"},
            {"App ID": "5", "Name": "Malformed", "Minutes Played": "not tracked"},
        ],
    )

    units = {unit.title: unit for unit in SteamLibraryCsvAdapter(path=str(export)).ingest(entity_types=["game"]).units}

    assert units["Zero"].metadata["playtime_minutes"] == 0
    assert units["Zero"].metadata["playtime_bucket"] == "unplayed"
    assert units["Short"].metadata["playtime_minutes"] == 59
    assert units["Short"].metadata["playtime_bucket"] == "sampled"
    assert units["Normal"].metadata["playtime_minutes"] == 60
    assert units["Normal"].metadata["playtime_bucket"] == "played"
    assert units["High"].metadata["playtime_minutes"] == 600
    assert units["High"].metadata["playtime_bucket"] == "deep_play"
    assert "playtime_minutes" not in units["Malformed"].metadata
    assert "playtime_bucket" not in units["Malformed"].metadata


def test_steam_library_csv_preserves_achievement_and_ownership_metadata(tmp_path):
    export = tmp_path / "steam.csv"
    _write_csv(
        export,
        [
            {
                "App ID": "730",
                "Name": "Counter-Strike 2",
                "Achievements Unlocked": "17",
                "Achievements Total": "33",
                "Completion Percent": "51.5%",
                "First Played": "2024-03-04",
                "Date Acquired": "2024-03-01T10:30:00-05:00",
                "Review Score": "88.5",
                "Owned Platforms": "Windows, macOS; Linux|Steam Deck",
            }
        ],
    )

    unit = SteamLibraryCsvAdapter(path=str(export)).ingest(entity_types=["game"]).units[0]

    assert unit.metadata["achievements_unlocked"] == 17
    assert unit.metadata["achievements_total"] == 33
    assert unit.metadata["completion_percent"] == 51.5
    assert unit.metadata["first_played"] == "2024-03-04T00:00:00+00:00"
    assert unit.metadata["date_acquired"] == "2024-03-01T15:30:00+00:00"
    assert unit.metadata["review_score"] == 88.5
    assert unit.metadata["owned_platforms"] == ["windows", "macos", "linux", "steam deck"]
    assert "Achievements: 17/33" in unit.content
    assert "Completion: 51.5%" in unit.content


def test_steam_library_csv_emits_genre_units_and_edges(tmp_path):
    export = tmp_path / "steam.csv"
    _write_csv(
        export,
        [
            {"App ID": "620", "Name": "Portal 2", "Hours Played": "12.5", "Last Played": "2025-01-02T03:04:05Z", "Genres": "Puzzle; Co-op"},
            {"App ID": "400", "Name": "Portal", "Minutes Played": "30", "Last Played": "2024-01-02", "Tags": "Puzzle"},
            {"App ID": "400", "Name": "Portal", "Minutes Played": "30", "Categories": "Puzzle|Single-player"},
        ],
    )

    result = SteamLibraryCsvAdapter(path=str(export)).ingest(entity_types=["game", "genre"])

    assert SteamLibraryCsvAdapter(path=str(export)).entity_types == ["game", "genre", "developer"]
    genres = sorted((unit for unit in result.units if unit.source_entity_type == "genre"), key=lambda unit: unit.title)
    assert [unit.title for unit in genres] == ["co-op", "puzzle", "single-player"]
    puzzle = next(unit for unit in genres if unit.title == "puzzle")
    games = [unit for unit in result.units if unit.source_entity_type == "game"]
    unique_game_source_ids = sorted({unit.source_id for unit in games})
    assert puzzle.metadata["game_count"] == 2
    assert puzzle.metadata["total_playtime_minutes"] == 780
    assert puzzle.metadata["game_source_ids"] == unique_game_source_ids
    assert puzzle.metadata["app_ids"] == ["400", "620"]
    assert puzzle.metadata["last_played_at"] == "2025-01-02T03:04:05+00:00"
    assert puzzle.metadata["source_files"] == ["steam.csv"]
    puzzle_edges = [edge for edge in result.edges if edge.from_unit_id == puzzle.source_id]
    assert [edge.to_unit_id for edge in puzzle_edges] == unique_game_source_ids
    assert len({edge.id for edge in puzzle_edges}) == len(puzzle_edges)

    genre_only = SteamLibraryCsvAdapter(path=str(export)).ingest(entity_types=["genre"])
    assert {unit.source_entity_type for unit in genre_only.units} == {"genre"}
    assert genre_only.edges == []


def test_steam_library_csv_emits_developer_units_and_edges(tmp_path):
    export = tmp_path / "steam.csv"
    _write_csv(
        export,
        [
            {
                "App ID": "620",
                "Name": "Portal 2",
                "Hours Played": "12.5",
                "Developer": "Valve",
                "Publishers": "Valve | Electronic Arts",
            },
            {
                "App ID": "400",
                "Name": "Portal",
                "Minutes Played": "30",
                "Developers": " valve ; Valve ",
                "Publisher": "Valve",
            },
        ],
    )

    result = SteamLibraryCsvAdapter(path=str(export)).ingest(entity_types=["game", "developer"])

    games = [unit for unit in result.units if unit.source_entity_type == "game"]
    developers = sorted((unit for unit in result.units if unit.source_entity_type == "developer"), key=lambda unit: unit.title)
    assert [unit.title for unit in developers] == ["Electronic Arts", "Valve"]
    valve = next(unit for unit in developers if unit.title == "Valve")
    assert games[0].metadata["creators"]
    assert valve.metadata["developer"] == "Valve"
    assert valve.metadata["game_count"] == 2
    assert valve.metadata["total_playtime_minutes"] == 780
    assert valve.metadata["game_source_ids"] == sorted(game.source_id for game in games)
    assert valve.metadata["app_ids"] == ["400", "620"]
    assert valve.metadata["source_files"] == ["steam.csv"]
    assert {edge.relation for edge in result.edges} == {EdgeRelation.CONTAINS}
    assert {edge.source for edge in result.edges} == {EdgeSource.SOURCE}
    assert {edge.to_unit_id for edge in result.edges if edge.from_unit_id == valve.source_id} == {game.source_id for game in games}


def test_steam_library_csv_developer_filtering(tmp_path):
    export = tmp_path / "steam.csv"
    _write_csv(export, [{"App ID": "1", "Name": "Game", "Developer": "Studio"}])

    developer_only = SteamLibraryCsvAdapter(path=str(export)).ingest(entity_types=["developer"])
    game_only = SteamLibraryCsvAdapter(path=str(export)).ingest(entity_types=["game"])

    assert [unit.source_entity_type for unit in developer_only.units] == ["developer"]
    assert developer_only.edges == []
    assert [unit.source_entity_type for unit in game_only.units] == ["game"]
    assert game_only.edges == []
