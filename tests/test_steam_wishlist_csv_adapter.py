from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.steam_wishlist_csv import SteamWishlistCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_steam_wishlist_csv_ingests_representative_rows(tmp_path):
    export = tmp_path / "wishlist.csv"
    _write_csv(
        export,
        [
            {
                "App ID": "620",
                "Title": "Portal 2",
                "Store URL": "https://store.steampowered.com/app/620/Portal_2/",
                "Release Date": "2011-04-18",
                "Review Score": "96%",
                "Review Count": "342,123",
                "Price": "$9.99",
                "Original Price": "$19.99",
                "Discount": "50%",
                "Tags": "Puzzle; Co-op",
                "Platforms": "Windows, macOS; Linux|Steam Deck",
                "Ranking": "2",
                "Wishlisted Date": "2026-05-01T09:00:00Z",
                "Notes": "Wait for sale",
            }
        ],
    )

    result = SteamWishlistCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "steam_wishlist_csv"
    assert unit.source_id == "steam_wishlist_csv:620"
    assert unit.source_entity_type == "wishlisted_game"
    assert unit.content_type == ContentType.METADATA
    assert unit.title == "Portal 2"
    assert unit.metadata["app_id"] == "620"
    assert unit.metadata["store_url"].endswith("/Portal_2/")
    assert unit.metadata["release_date"] == "2011-04-18T00:00:00+00:00"
    assert unit.metadata["review_score"] == 96.0
    assert unit.metadata["review_count"] == 342123
    assert unit.metadata["price"] == 9.99
    assert unit.metadata["original_price"] == 19.99
    assert unit.metadata["discount_percent"] == 50
    assert unit.metadata["genres"] == ["puzzle", "co-op"]
    assert unit.metadata["platforms"] == ["windows", "macos", "linux", "steam deck"]
    assert unit.metadata["ranking"] == 2
    assert unit.metadata["added_at"] == "2026-05-01T09:00:00+00:00"
    assert unit.metadata["source_file"] == "wishlist.csv"
    assert unit.metadata["source_row"]["Title"] == "Portal 2"
    assert unit.created_at == datetime(2026, 5, 1, 9, tzinfo=timezone.utc)
    assert {"steam", "wishlist", "puzzle", "co-op", "windows", "steam deck"}.issubset(set(unit.tags))


def test_steam_wishlist_csv_sparse_rows_fallback_ids_and_generated_urls(tmp_path):
    export = tmp_path / "wishlist.csv"
    _write_csv(
        export,
        [
            {"appid": "400", "game": "Portal", "Genres": "Puzzle"},
            {"Name": "No App Id Game", "Added": "2026-05-02", "Platforms": "Windows"},
            {"Name": ""},
        ],
    )

    result = SteamWishlistCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    units = {unit.title: unit for unit in result.units}
    first = units["Portal"]
    assert first.source_id == "steam_wishlist_csv:400"
    assert first.metadata["store_url"] == "https://store.steampowered.com/app/400/"
    assert first.metadata["genres"] == ["puzzle"]
    second = units["No App Id Game"]
    assert second.title == "No App Id Game"
    assert second.source_id.startswith("steam_wishlist_csv:")
    repeated = {unit.title: unit for unit in SteamWishlistCsvAdapter(path=str(export)).ingest().units}
    assert second.source_id == repeated["No App Id Game"].source_id
    assert second.metadata["platforms"] == ["windows"]


def test_steam_wishlist_csv_directory_sorts_filters_since_and_entity_types(tmp_path):
    old = tmp_path / "old.csv"
    _write_csv(old, [{"App ID": "1", "Title": "Old", "Wishlisted Date": "2026-05-01"}])
    new = tmp_path / "new.csv"
    _write_csv(new, [{"App ID": "2", "Title": "New", "Wishlisted Date": "2026-05-03"}])
    ignored = tmp_path / "ignored.txt"
    ignored.write_text("App ID,Title\n3,Ignored\n", encoding="utf-8")
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff\xfe\x00")
    since = SyncState(source_project="steam_wishlist_csv", source_entity_type="wishlisted_game", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = SteamWishlistCsvAdapter(path=str(tmp_path)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["steam_wishlist_csv:2"]
    assert SteamWishlistCsvAdapter(path=str(tmp_path)).ingest(entity_types=["game"]).units == []
    all_units = SteamWishlistCsvAdapter(path=str(tmp_path)).ingest().units
    assert [unit.source_id for unit in all_units] == ["steam_wishlist_csv:1", "steam_wishlist_csv:2"]
    assert [(unit.updated_at, unit.source_id) for unit in all_units] == sorted((unit.updated_at, unit.source_id) for unit in all_units)
