from __future__ import annotations

import json

from graph.adapters.google_maps_timeline_json import GoogleMapsTimelineJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, SourceProject


def test_google_maps_timeline_json_ingests_visits_segments_and_links_chronologically(tmp_path):
    export = tmp_path / "timeline.json"
    export.write_text(
        json.dumps(
            {
                "timelineObjects": [
                    {
                        "placeVisit": {
                            "location": {
                                "name": "Coffee Shop",
                                "address": "1 Main St",
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

    result = GoogleMapsTimelineJsonAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["place_visit", "activity_segment"]
    assert result.units[0].source_project == SourceProject.GOOGLE_MAPS_TIMELINE_JSON
    assert result.units[0].metadata["place_name"] == "Coffee Shop"
    assert result.units[0].metadata["latitude"] == 35.5
    assert result.units[1].metadata["activity_type"] == "walking"
    assert result.units[1].metadata["distance_meters"] == 1200.0
    assert len(result.edges) == 1
    assert result.edges[0].relation == EdgeRelation.REFERENCES
    assert result.edges[0].from_unit_id == result.units[0].source_id
    assert result.edges[0].to_unit_id == result.units[1].source_id
    assert get_adapter("google_maps_timeline_json", path=str(export)).name == "google_maps_timeline_json"
