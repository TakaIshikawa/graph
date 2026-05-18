from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.google_location_semantic_history_json import GoogleLocationSemanticHistoryJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_google_location_semantic_history_json_ingests_places_and_activities(tmp_path):
    export = tmp_path / "Semantic Location History.json"
    export.write_text(
        json.dumps(
            {
                "timelineObjects": [
                    {
                        "placeVisit": {
                            "location": {
                                "name": "Coffee Shop",
                                "address": "1 Main St, Tokyo",
                                "placeId": "abc123",
                                "latitudeE7": 355000000,
                                "longitudeE7": 1397000000,
                            },
                            "duration": {
                                "startTimestamp": "2025-01-01T09:00:00Z",
                                "endTimestamp": "2025-01-01T10:00:00Z",
                            },
                            "placeConfidence": "HIGH_CONFIDENCE",
                        }
                    },
                    {
                        "activitySegment": {
                            "activityType": "WALKING",
                            "distance": 1200,
                            "confidence": "HIGH",
                            "duration": {
                                "startTimestamp": "2025-01-01T10:00:00Z",
                                "endTimestamp": "2025-01-01T10:20:00Z",
                            },
                            "startLocation": {"latitudeE7": 355000000, "longitudeE7": 1397000000},
                            "endLocation": {"latitudeE7": 355100000, "longitudeE7": 1397100000},
                        }
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = GoogleLocationSemanticHistoryJsonAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["place_visit", "activity_segment"]
    place, activity = result.units
    assert place.source_project == SourceProject.GOOGLE_LOCATION_SEMANTIC_HISTORY_JSON
    assert place.title == "Coffee Shop"
    assert "Place visit: Coffee Shop" in place.content
    assert "Address: 1 Main St, Tokyo" in place.content
    assert "Time range: 2025-01-01T09:00:00+00:00 to 2025-01-01T10:00:00+00:00" in place.content
    assert place.metadata["place_name"] == "Coffee Shop"
    assert place.metadata["address"] == "1 Main St, Tokyo"
    assert place.metadata["confidence"] == "HIGH_CONFIDENCE"
    assert place.metadata["latitude"] == 35.5
    assert place.metadata["longitude"] == 139.7
    assert place.metadata["source_file"] == "Semantic Location History.json"
    assert place.metadata["record_index"] == 0

    assert activity.title == "walking"
    assert "Activity segment: walking" in activity.content
    assert "Distance: 1200 meters" in activity.content
    assert "Time range: 2025-01-01T10:00:00+00:00 to 2025-01-01T10:20:00+00:00" in activity.content
    assert activity.metadata["activity_type"] == "walking"
    assert activity.metadata["distance_meters"] == 1200.0
    assert activity.metadata["confidence"] == "HIGH"
    assert activity.metadata["start_latitude"] == 35.5
    assert activity.metadata["end_longitude"] == 139.71


def test_google_location_semantic_history_json_supports_registry_filters_since_and_bad_files(tmp_path):
    export = tmp_path / "history.json"
    export.write_text(
        json.dumps(
            {
                "timelineObjects": [
                    {
                        "placeVisit": {
                            "location": {"name": "Old Place", "address": "Old Address"},
                            "duration": {
                                "startTimestampMs": "1735689600000",
                                "endTimestampMs": "1735693200000",
                            },
                        }
                    },
                    {
                        "activitySegment": {
                            "activities": [{"activityType": "IN_PASSENGER_VEHICLE"}],
                            "distance": "5000",
                            "duration": {
                                "startTimestamp": "2025-01-02T00:00:00Z",
                                "endTimestamp": "2025-01-02T01:00:00Z",
                            },
                        }
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    adapter = get_adapter("google_location_semantic_history_json", path=str(tmp_path))
    result = adapter.ingest(
        since=SyncState(
            source_project="google_location_semantic_history_json",
            source_entity_type="activity_segment",
            last_sync_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        ),
        entity_types=["activity_segment"],
    )

    assert isinstance(adapter, GoogleLocationSemanticHistoryJsonAdapter)
    assert adapter.name == "google_location_semantic_history_json"
    assert [unit.source_entity_type for unit in result.units] == ["activity_segment"]
    assert result.units[0].metadata["activity_type"] == "in_passenger_vehicle"
    assert result.units[0].metadata["distance_meters"] == 5000.0
    assert GoogleLocationSemanticHistoryJsonAdapter(path=str(tmp_path / "missing.json")).ingest().units == []
