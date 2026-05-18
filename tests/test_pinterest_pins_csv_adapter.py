from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.pinterest_pins_csv import PinterestPinsCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_pinterest_pins_csv_ingests_pin_metadata_and_bookmark_tags(tmp_path):
    export = tmp_path / "pins.csv"
    export.write_text(
        "title,description,board,saved_at,link,image_url,pin_url\n"
        "Desk Setup,Standing desk inspiration,Workspaces,2026-05-01T10:00:00Z,https://example.com/desk,https://i.pinimg.com/desk.jpg,https://www.pinterest.com/pin/123/\n"
        "Garden Plan,,Garden,2026-05-02,https://example.com/garden,https://i.pinimg.com/garden.jpg,https://www.pinterest.com/pin/456/\n",
        encoding="utf-8",
    )

    result = PinterestPinsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    first = result.units[0]
    assert first.source_project == "pinterest_pins_csv"
    assert first.source_entity_type == "pin"
    assert first.content_type == ContentType.ARTIFACT
    assert first.title == "Desk Setup"
    assert first.metadata["title"] == "Desk Setup"
    assert first.metadata["description"] == "Standing desk inspiration"
    assert first.metadata["board"] == "Workspaces"
    assert first.metadata["saved_at"] == "2026-05-01T10:00:00+00:00"
    assert first.metadata["link"] == "https://example.com/desk"
    assert first.metadata["url"] == "https://example.com/desk"
    assert first.metadata["source_url"] == "https://example.com/desk"
    assert first.metadata["external_url"] == "https://example.com/desk"
    assert first.metadata["image_url"] == "https://i.pinimg.com/desk.jpg"
    assert first.metadata["pin_url"] == "https://www.pinterest.com/pin/123/"
    assert {"pinterest", "pin", "bookmark", "workspaces"}.issubset(set(first.tags))
    assert "Description: Standing desk inspiration" in first.content
    assert "Board: Workspaces" in first.content
    assert "URL: https://example.com/desk" in first.content
    assert "Pin: https://www.pinterest.com/pin/123/" in first.content


def test_pinterest_pins_csv_supports_alternate_headers_and_blank_rows(tmp_path):
    export = tmp_path / "pins.csv"
    export.write_text(
        "Title,Description,Board Name,Created At,Source URL,Image URL,Pinterest URL\n"
        ",,,,,,\n"
        "Recipe,Soup reference,Food,2026-05-03,https://example.com/soup,https://i.pinimg.com/soup.jpg,https://www.pinterest.com/pin/789/\n",
        encoding="utf-8",
    )

    result = PinterestPinsCsvAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["Recipe"]
    assert result.units[0].metadata["board"] == "Food"
    assert result.units[0].metadata["pin_url"] == "https://www.pinterest.com/pin/789/"
    assert "Soup reference" in result.units[0].content


def test_pinterest_pins_csv_has_stable_ids_and_filters_since(tmp_path):
    export = tmp_path / "pins.csv"
    export.write_text(
        "title,board,saved_at,link,pin_url\n"
        "Old,Archive,2026-05-01,https://example.com/old,https://www.pinterest.com/pin/old/\n"
        "New,Archive,2026-05-03,https://example.com/new,https://www.pinterest.com/pin/new/\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="pinterest_pins_csv",
        source_entity_type="pin",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    first = PinterestPinsCsvAdapter(path=str(export)).ingest().units
    second = PinterestPinsCsvAdapter(path=str(export)).ingest().units
    filtered = PinterestPinsCsvAdapter(path=str(export)).ingest(since=since).units

    assert [unit.source_id for unit in first] == [unit.source_id for unit in second]
    assert [unit.title for unit in filtered] == ["New"]
    assert PinterestPinsCsvAdapter(path=str(export)).ingest(entity_types=["bookmark"]).units == []
