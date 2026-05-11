from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.day_one_json import DayOneJsonAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_day_one_json_ingests_entries_with_metadata(tmp_path):
    export = tmp_path / "Journal.json"
    export.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "uuid": "entry-1",
                        "text": "Morning run\nFelt good.",
                        "creationDate": "2025-01-01T08:00:00Z",
                        "modifiedDate": "2025-01-01T09:00:00Z",
                        "tags": ["health", "reflection"],
                        "location": {"latitude": 35.0, "longitude": 139.0, "placeName": "Tokyo"},
                        "weather": {"temperatureCelsius": 10, "conditionsDescription": "Sunny"},
                        "starred": True,
                        "photos": [{"identifier": "photo-1", "md5": "abc"}],
                        "audio": [{"identifier": "audio-1", "duration": 12}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = DayOneJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.DAY_ONE_JSON
    assert unit.source_id == "day_one_json:entry-1"
    assert unit.source_entity_type == "entry"
    assert unit.title == "Morning run"
    assert unit.content == "Morning run\nFelt good."
    assert unit.created_at == datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)
    assert unit.metadata["tags"] == ["health", "reflection"]
    assert unit.metadata["location"]["placeName"] == "Tokyo"
    assert unit.metadata["weather"]["conditionsDescription"] == "Sunny"
    assert unit.metadata["starred"] is True
    assert unit.metadata["attachments"]["photos"][0]["identifier"] == "photo-1"
    assert unit.metadata["attachments"]["audio"][0]["identifier"] == "audio-1"
    assert "health" in unit.tags


def test_day_one_json_reads_export_directory_and_skips_entries_without_uuid(tmp_path):
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    (tmp_path / "entries.json").write_text(
        json.dumps(
            [
                {"text": "No UUID", "creationDate": "2025-01-01T00:00:00Z"},
                {"uuid": "entry-2", "text": "Valid", "creationDate": "2025-01-02T00:00:00Z"},
            ]
        ),
        encoding="utf-8",
    )

    result = DayOneJsonAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id == "day_one_json:entry-2"


def test_day_one_json_adapter_is_registered():
    assert "day_one_json" in list_adapters()
    assert get_adapter("day_one_json", path="/tmp/dayone").name == "day_one_json"
