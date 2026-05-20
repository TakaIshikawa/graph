from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.google_takeout_saved_places_json import GoogleTakeoutSavedPlacesJsonAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_google_takeout_saved_places_json_ingests_feature_collection_metadata(tmp_path):
    export = tmp_path / "saved-places.json"
    export.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "title": "Blue Bottle Coffee",
                            "address": "1 Ferry Building, San Francisco, CA",
                            "url": "https://maps.google.com/?cid=123",
                            "place_id": "ChIJ123",
                            "list_name": "Starred places",
                            "notes": "Near the water",
                            "saved_at": "2025-02-03T04:05:06Z",
                        },
                        "geometry": {"type": "Point", "coordinates": [-122.393, 37.795]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = GoogleTakeoutSavedPlacesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "google_takeout_saved_places_json"
    assert unit.source_id.startswith("google_takeout_saved_places_json:")
    assert unit.source_entity_type == "saved_place"
    assert unit.title == "Blue Bottle Coffee"
    assert unit.content_type == ContentType.METADATA
    assert "Place: Blue Bottle Coffee" in unit.content
    assert "Address: 1 Ferry Building, San Francisco, CA" in unit.content
    assert "List: Starred places" in unit.content
    assert "Notes: Near the water" in unit.content
    assert "Place ID: ChIJ123" in unit.content
    assert "Coordinates: 37.795, -122.393" in unit.content
    assert "URL: https://maps.google.com/?cid=123" in unit.content
    assert unit.metadata["name"] == "Blue Bottle Coffee"
    assert unit.metadata["address"] == "1 Ferry Building, San Francisco, CA"
    assert unit.metadata["url"] == "https://maps.google.com/?cid=123"
    assert unit.metadata["place_id"] == "ChIJ123"
    assert unit.metadata["list_name"] == "Starred places"
    assert unit.metadata["notes"] == "Near the water"
    assert unit.metadata["latitude"] == 37.795
    assert unit.metadata["longitude"] == -122.393
    assert unit.metadata["saved_at"] == "2025-02-03T04:05:06+00:00"
    assert unit.metadata["source_file"] == "saved-places.json"
    assert unit.metadata["record_index"] == 0
    assert unit.tags == ["google", "saved-place", "starred-places"]
    assert unit.created_at == datetime(2025, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 2, 3, 4, 5, 6, tzinfo=timezone.utc)


def test_google_takeout_saved_places_json_skips_invalid_records_and_filters_since(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "places.json").write_text(
        json.dumps(
            [
                {},
                {
                    "name": "Old Place",
                    "address": "1 Old St",
                    "created_at": "2025-01-01T00:00:00Z",
                    "coordinates": {"lat": 1.0, "lng": 2.0},
                },
                {
                    "name": "New Place",
                    "address": "2 New St",
                    "created_at": "2025-01-03T00:00:00Z",
                    "coordinates": {"latitude": 3.0, "longitude": 4.0},
                },
                {
                    "name": "No Date Place",
                    "url": "https://maps.google.com/no-date",
                    "list_name": "Want to go",
                },
            ]
        ),
        encoding="utf-8",
    )

    adapter = GoogleTakeoutSavedPlacesJsonAdapter(path=str(tmp_path))
    sync = SyncState(
        source_project="google_takeout_saved_places_json",
        source_entity_type="saved_place",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    result = adapter.ingest(since=sync)

    assert sorted(unit.metadata["name"] for unit in result.units) == ["New Place", "No Date Place"]
    new_place = next(unit for unit in result.units if unit.metadata["name"] == "New Place")
    assert new_place.metadata["latitude"] == 3.0
    assert new_place.metadata["longitude"] == 4.0
    no_date = next(unit for unit in result.units if unit.metadata["name"] == "No Date Place")
    assert no_date.metadata["list_name"] == "Want to go"
    assert len({unit.source_id for unit in result.units}) == 2
    assert adapter.ingest(entity_types=["place_visit"]).units == []
