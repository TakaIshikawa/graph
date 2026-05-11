from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.steam_library_csv import SteamLibraryCsvAdapter
from graph.types.enums import SourceProject


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

    result = SteamLibraryCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.STEAM_LIBRARY_CSV
    assert unit.source_id == "steam_library_csv:620"
    assert unit.title == "Portal 2"
    assert unit.metadata["app_id"] == "620"
    assert unit.metadata["playtime_minutes"] == 750
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

    result = SteamLibraryCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    first = result.units[0]
    assert first.source_id == "steam_library_csv:400"
    assert first.metadata["playtime_minutes"] == 123
    assert first.metadata["store_url"] == "https://store.steampowered.com/app/400/"
    assert "last_played" not in first.metadata
    assert result.units[1].title == "No App Id Game"
    assert get_adapter("steam_library_csv", path=str(export)).name == "steam_library_csv"
