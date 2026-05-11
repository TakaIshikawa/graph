from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.activitywatch_json import ActivityWatchJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_activitywatch_json_ingests_browser_window_and_afk_events(tmp_path):
    export = tmp_path / "aw.json"
    export.write_text(
        json.dumps(
            {
                "buckets": {
                    "aw-watcher-web": {
                        "type": "web.tab.current",
                        "events": [
                            {
                                "timestamp": "2025-01-01T10:00:00Z",
                                "duration": 30.5,
                                "data": {"app": "Firefox", "title": "Docs", "url": "https://example.com"},
                            }
                        ],
                    },
                    "aw-watcher-window": {
                        "type": "currentwindow",
                        "events": [
                            {"timestamp": "2025-01-01T10:01:00+00:00", "duration": 60, "data": {"app": "Code", "title": "main.py"}}
                        ],
                    },
                    "aw-watcher-afk": {
                        "type": "afkstatus",
                        "events": [{"timestamp": "2025-01-01T10:02:00Z", "duration": 120, "data": {"status": "afk"}}],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = ActivityWatchJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 3
    browser = next(unit for unit in result.units if unit.metadata["bucket_id"] == "aw-watcher-web")
    assert browser.source_project == SourceProject.ACTIVITYWATCH_JSON
    assert browser.metadata["bucket_type"] == "web.tab.current"
    assert browser.metadata["app"] == "Firefox"
    assert browser.metadata["url"] == "https://example.com"
    assert browser.metadata["duration"] == 30.5
    assert browser.metadata["data"]["title"] == "Docs"
    assert "activitywatch" in browser.tags
    assert "web.tab.current" in browser.tags
    assert browser.source_id == ActivityWatchJsonAdapter(path=str(export)).ingest().units[0].source_id


def test_activitywatch_json_filters_and_registry(tmp_path):
    export = tmp_path / "aw.json"
    export.write_text(
        json.dumps(
            {
                "id": "bucket",
                "type": "currentwindow",
                "events": [
                    {"timestamp": "2025-01-01T00:00:00Z", "duration": 1, "data": {"app": "Old"}},
                    {"timestamp": "2025-02-01T00:00:00Z", "duration": 1, "data": {"app": "New"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="activitywatch_json",
        source_entity_type="activity",
        last_sync_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
    )

    result = ActivityWatchJsonAdapter(path=str(export)).ingest(since=since)

    assert [unit.metadata["app"] for unit in result.units] == ["New"]
    assert ActivityWatchJsonAdapter(path=str(export)).ingest(entity_types=["event"]).units == []
    assert get_adapter("activitywatch_json", path=str(export)).name == "activitywatch_json"
