from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.steam_playtime_csv import SteamPlaytimeCsvAdapter
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_steam_playtime_csv_extracts_metadata_and_numbers(tmp_path):
    export = tmp_path / "playtime.csv"
    _write_csv(
        export,
        [
            {
                "App ID": "620",
                "Game Name": "Portal 2",
                "Hours Played": "12.5 hours",
                "Last Played": "2025-01-02T03:04:05Z",
                "First Played": "2024-12-01",
                "Achievements Unlocked": "17",
                "Achievements Total": "51",
                "Store URL": "https://store.steampowered.com/app/620/Portal_2/",
                "Platforms": "Windows; Steam Deck",
            }
        ],
    )

    unit = SteamPlaytimeCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "steam_playtime_csv"
    assert unit.source_id == "steam_playtime_csv:620"
    assert unit.source_entity_type == "game_playtime"
    assert unit.title == "Portal 2"
    assert unit.metadata["hours_played"] == 12.5
    assert unit.metadata["achievements_unlocked"] == 17
    assert unit.metadata["achievements_total"] == 51
    assert unit.metadata["last_played"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["first_played"] == "2024-12-01T00:00:00+00:00"
    assert unit.metadata["platforms"] == ["windows", "steam deck"]
    assert "Achievements: 17/51" in unit.content


def test_steam_playtime_csv_stable_fallback_ids_and_blank_rows(tmp_path):
    export = tmp_path / "playtime.csv"
    _write_csv(export, [{"Name": "No App Id Game", "Minutes Played": "90"}, {"Name": "", "Hours Played": ""}])

    units = SteamPlaytimeCsvAdapter(path=str(export)).ingest().units

    assert len(units) == 1
    assert units[0].source_id == SteamPlaytimeCsvAdapter()._source_id("", "No App Id Game")
    assert units[0].metadata["hours_played"] == 1.5
    assert units[0].metadata["minutes_played"] == 90


def test_steam_playtime_csv_sorts_and_filters_since_and_entity_type(tmp_path):
    export = tmp_path / "playtime.csv"
    _write_csv(
        export,
        [
            {"App ID": "2", "Name": "New", "Last Played": "2025-01-03"},
            {"App ID": "1", "Name": "Old", "Last Played": "2025-01-01"},
            {"App ID": "3", "Name": "Same Day", "Last Played": "2025-01-03"},
        ],
    )

    units = SteamPlaytimeCsvAdapter(path=str(export)).ingest().units
    assert [unit.source_id for unit in units] == ["steam_playtime_csv:1", "steam_playtime_csv:2", "steam_playtime_csv:3"]

    since = SyncState(
        source_project="steam_playtime_csv",
        source_entity_type="game_playtime",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    filtered = SteamPlaytimeCsvAdapter(path=str(export)).ingest(since=since).units
    assert [unit.title for unit in filtered] == ["New", "Same Day"]
    assert SteamPlaytimeCsvAdapter(path=str(export)).ingest(entity_types=["game"]).units == []
