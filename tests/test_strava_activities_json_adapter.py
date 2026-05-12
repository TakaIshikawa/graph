from __future__ import annotations

import json

from graph.adapters.registry import get_adapter
from graph.adapters.strava_activities_json import StravaActivitiesJsonAdapter
from graph.types.enums import SourceProject


def test_strava_activities_json_ingests_array_and_directory(tmp_path):
    (tmp_path / "activities.json").write_text(json.dumps([{"id": 1, "name": "Morning Run", "sport_type": "Run", "start_date": "2026-05-01T00:00:00Z", "distance": 5000, "moving_time": 1500, "total_elevation_gain": 42, "average_speed": 3.3, "kudos_count": 5}]), encoding="utf-8")
    (tmp_path / "single.json").write_text(json.dumps({"id": 2, "name": "Ride", "type": "Ride", "start_date": "2026-05-02T00:00:00Z"}), encoding="utf-8")

    result = StravaActivitiesJsonAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == ["strava_activities_json:1", "strava_activities_json:2"]
    assert result.units[0].source_project == SourceProject.STRAVA_ACTIVITIES_JSON
    assert result.units[0].metadata["distance"] == 5000.0
    assert "Run" in result.units[0].tags
    assert get_adapter("strava_activities_json", path=str(tmp_path)).name == "strava_activities_json"
