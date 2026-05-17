from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.google_location_history_semantic_json import GoogleLocationHistorySemanticJsonAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_google_location_history_semantic_json_ingests_place_visits_and_activity_segments(tmp_path):
    export = tmp_path / "semantic.json"
    export.write_text(
        json.dumps(
            {
                "timelineObjects": [
                    {
                        "placeVisit": {
                            "location": {
                                "name": "Coffee Shop",
                                "address": "1 Main St",
                                "placeId": "place-1",
                                "latitudeE7": 355000000,
                                "longitudeE7": 1397000000,
                            },
                            "duration": {
                                "startTimestamp": "2026-05-01T09:00:00Z",
                                "endTimestamp": "2026-05-01T10:00:00Z",
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
                                "startTimestamp": "2026-05-01T10:00:00Z",
                                "endTimestamp": "2026-05-01T10:20:00Z",
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

    result = GoogleLocationHistorySemanticJsonAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["place_visit", "activity_segment"]
    place = result.units[0]
    assert place.source_project == "google_location_history_semantic_json"
    assert place.content_type == ContentType.METADATA
    assert place.title == "Coffee Shop"
    assert place.metadata["place_name"] == "Coffee Shop"
    assert place.metadata["address"] == "1 Main St"
    assert place.metadata["place_id"] == "place-1"
    assert place.metadata["start_at"] == "2026-05-01T09:00:00+00:00"
    assert place.metadata["end_at"] == "2026-05-01T10:00:00+00:00"
    assert place.metadata["latitude"] == 35.5
    assert place.metadata["longitude"] == 139.7
    assert place.metadata["confidence"] == "HIGH_CONFIDENCE"
    assert {"google", "location-history", "place_visit"}.issubset(set(place.tags))

    segment = result.units[1]
    assert segment.title == "Walking"
    assert segment.metadata["activity_type"] == "walking"
    assert segment.metadata["distance_meters"] == 1200.0
    assert segment.metadata["start_latitude"] == 35.5
    assert segment.metadata["start_longitude"] == 139.7
    assert segment.metadata["end_latitude"] == 35.51
    assert segment.metadata["end_longitude"] == 139.71
    assert segment.metadata["confidence"] == "HIGH"


def test_google_location_history_semantic_json_handles_optional_fields_and_filters_since(tmp_path):
    export = tmp_path / "semantic.json"
    export.write_text(
        json.dumps(
            {
                "timelineObjects": [
                    {
                        "placeVisit": {
                            "location": {"name": "Old Place"},
                            "duration": {"startTimestamp": "2026-05-01T09:00:00Z", "endTimestamp": "2026-05-01T10:00:00Z"},
                        }
                    },
                    {
                        "placeVisit": {
                            "location": {"address": "2 Main St", "latLng": "35.6, 139.8"},
                            "duration": {"startTimestamp": "2026-05-03T09:00:00Z"},
                        }
                    },
                    {"activitySegment": {"activities": [{"activityType": "CYCLING"}], "duration": {"startTimestampMs": "1777812000000"}}},
                ]
            }
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="google_location_history_semantic_json",
        source_entity_type="place_visit",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    result = GoogleLocationHistorySemanticJsonAdapter(path=str(export)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["2 Main St", "Cycling"]
    assert result.units[0].metadata["latitude"] == 35.6
    assert result.units[0].metadata["longitude"] == 139.8
    assert "place_id" not in result.units[0].metadata
    assert "None" not in result.units[0].content
    assert GoogleLocationHistorySemanticJsonAdapter(path=str(export)).ingest(entity_types=["place_visit"]).units[0].source_entity_type == "place_visit"


def test_google_location_history_semantic_json_source_ids_are_stable(tmp_path):
    export = tmp_path / "semantic.json"
    export.write_text(
        json.dumps(
            {
                "timelineObjects": [
                    {
                        "placeVisit": {
                            "location": {"name": "Stable Place", "placeId": "stable-place"},
                            "duration": {"startTimestamp": "2026-05-01T09:00:00Z"},
                        }
                    },
                    {
                        "activitySegment": {
                            "activityType": "RUNNING",
                            "distance": 500,
                            "duration": {"startTimestamp": "2026-05-01T10:00:00Z"},
                        }
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    first = GoogleLocationHistorySemanticJsonAdapter(path=str(export)).ingest().units
    second = GoogleLocationHistorySemanticJsonAdapter(path=str(export)).ingest().units

    assert [unit.source_id for unit in first] == [unit.source_id for unit in second]
    assert all(unit.source_id.startswith("google_location_history_semantic_json:") for unit in first)
