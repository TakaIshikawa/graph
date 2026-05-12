from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.activitywatch_json import ActivityWatchJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
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
                                "data": {
                                    "app": "Firefox",
                                    "title": "Docs",
                                    "url": "https://example.com",
                                },
                            }
                        ],
                    },
                    "aw-watcher-window": {
                        "type": "currentwindow",
                        "events": [
                            {
                                "timestamp": "2025-01-01T10:01:00+00:00",
                                "duration": 60,
                                "data": {"app": "Code", "title": "main.py"},
                            }
                        ],
                    },
                    "aw-watcher-afk": {
                        "type": "afkstatus",
                        "events": [
                            {
                                "timestamp": "2025-01-01T10:02:00Z",
                                "duration": 120,
                                "data": {"status": "afk"},
                            }
                        ],
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
    assert browser.metadata["normalized_url"] == "https://example.com"
    assert browser.metadata["domain"] == "example.com"
    assert browser.metadata["duration"] == 30.5
    assert browser.metadata["data"]["title"] == "Docs"
    assert "activitywatch" in browser.tags
    assert "web.tab.current" in browser.tags
    assert "domain:example.com" in browser.tags
    assert (
        browser.source_id == ActivityWatchJsonAdapter(path=str(export)).ingest().units[0].source_id
    )

    afk = next(unit for unit in result.units if unit.metadata["bucket_id"] == "aw-watcher-afk")
    assert afk.title == "ActivityWatch AFK"
    assert afk.metadata == {
        "bucket_id": "aw-watcher-afk",
        "bucket_type": "afkstatus",
        "status": "afk",
        "duration": 120.0,
        "timestamp": "2025-01-01T10:02:00+00:00",
        "source_file": "aw.json",
    }
    assert afk.tags == ["activitywatch", "afk"]


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


def test_activitywatch_json_filters_afk_events_since(tmp_path):
    export = tmp_path / "aw.json"
    export.write_text(
        json.dumps(
            {
                "id": "aw-watcher-afk",
                "type": "afkstatus",
                "events": [
                    {
                        "timestamp": "2025-01-01T00:00:00Z",
                        "duration": "10.5",
                        "data": {"status": "afk"},
                    },
                    {
                        "timestamp": "2025-01-02T00:00:00Z",
                        "duration": "20",
                        "data": {"status": "not-afk"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ActivityWatchJsonAdapter(path=str(export)).ingest(
        since=SyncState(
            source_project="activitywatch_json",
            source_entity_type="activity",
            last_sync_at=datetime(2025, 1, 1, 12, tzinfo=timezone.utc),
        )
    )

    assert [(unit.metadata["status"], unit.metadata["duration"]) for unit in result.units] == [
        ("not-afk", 20.0)
    ]


def test_activitywatch_json_ignores_malformed_afk_events_and_sorts_deterministically(tmp_path):
    export = tmp_path / "aw.json"
    export.write_text(
        json.dumps(
            {
                "buckets": {
                    "aw-watcher-afk": {
                        "type": "afkstatus",
                        "events": [
                            {"timestamp": "bad", "duration": 1, "data": {"status": "afk"}},
                            {
                                "timestamp": "2025-01-01T00:00:02Z",
                                "duration": "bad",
                                "data": {"status": "not-afk"},
                            },
                            {
                                "timestamp": "2025-01-01T00:00:01Z",
                                "duration": 1,
                                "data": {"status": "unknown"},
                            },
                            {
                                "timestamp": "2025-01-01T00:00:00Z",
                                "duration": 2,
                                "data": {"status": "afk"},
                            },
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = ActivityWatchJsonAdapter(path=str(export)).ingest()

    assert [unit.metadata["timestamp"] for unit in result.units] == [
        "2025-01-01T00:00:00+00:00",
        "2025-01-01T00:00:02+00:00",
    ]
    assert result.units[1].metadata["duration"] is None


def test_activitywatch_json_adds_normalized_http_domain_metadata(tmp_path):
    export = tmp_path / "aw.json"
    export.write_text(
        json.dumps(
            {
                "id": "web",
                "type": "web.tab.current",
                "events": [
                    {
                        "timestamp": "2025-01-01T00:00:00Z",
                        "duration": 1,
                        "data": {
                            "title": "Fragmented",
                            "url": "HTTPS://Example.COM:443/path/page?x=1&Y=Two#section",
                        },
                    },
                    {
                        "timestamp": "2025-01-01T00:00:01Z",
                        "duration": 1,
                        "data": {
                            "title": "Default HTTP port",
                            "url": "http://Example.COM:80/a?b=c#ignored",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ActivityWatchJsonAdapter(path=str(export)).ingest()

    https_unit, http_unit = result.units
    assert https_unit.metadata["url"] == "HTTPS://Example.COM:443/path/page?x=1&Y=Two#section"
    assert https_unit.metadata["normalized_url"] == "https://example.com/path/page?x=1&Y=Two"
    assert https_unit.metadata["domain"] == "example.com"
    assert https_unit.tags.count("domain:example.com") == 1
    assert http_unit.metadata["normalized_url"] == "http://example.com/a?b=c"
    assert http_unit.metadata["domain"] == "example.com"


def test_activitywatch_json_omits_domain_for_missing_and_non_http_urls(tmp_path):
    export = tmp_path / "aw.json"
    export.write_text(
        json.dumps(
            {
                "buckets": {
                    "web": {
                        "type": "web.tab.current",
                        "events": [
                            {
                                "timestamp": "2025-01-01T00:00:00Z",
                                "data": {"title": "No URL"},
                            },
                            {
                                "timestamp": "2025-01-01T00:00:01Z",
                                "data": {
                                    "title": "File URL",
                                    "url": "file:///Users/taka/report.html#local",
                                },
                            },
                        ],
                    },
                    "window": {
                        "type": "currentwindow",
                        "events": [
                            {
                                "timestamp": "2025-01-01T00:00:02Z",
                                "data": {"app": "Code", "title": "main.py"},
                            }
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = ActivityWatchJsonAdapter(path=str(export)).ingest()

    no_url, file_url, window = result.units
    assert "domain" not in no_url.metadata
    assert "normalized_url" not in no_url.metadata
    assert file_url.metadata["url"] == "file:///Users/taka/report.html#local"
    assert "domain" not in file_url.metadata
    assert "normalized_url" not in file_url.metadata
    assert "domain" not in window.metadata
    assert all(not tag.startswith("domain:") for tag in no_url.tags + file_url.tags + window.tags)


def test_activitywatch_json_emits_transition_edges_for_sorted_units(tmp_path):
    export = tmp_path / "aw.json"
    export.write_text(
        json.dumps(
            {
                "id": "window",
                "type": "currentwindow",
                "events": [
                    {
                        "timestamp": "2025-01-01T10:00:00Z",
                        "duration": 60,
                        "data": {"app": "Code", "title": "main.py"},
                    },
                    {
                        "timestamp": "2025-01-01T10:01:00Z",
                        "duration": 120,
                        "data": {"app": "Firefox", "title": "Docs"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ActivityWatchJsonAdapter(path=str(export)).ingest()

    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_unit_id == result.units[0].source_id
    assert edge.to_unit_id == result.units[1].source_id
    assert edge.relation == EdgeRelation.RELATES_TO
    assert edge.source == EdgeSource.SOURCE
    assert edge.metadata["elapsed_seconds"] == 60
    assert edge.metadata["previous_app"] == "Code"
    assert edge.metadata["next_app"] == "Firefox"
    assert edge.metadata["previous_title"] == "main.py"
    assert edge.metadata["next_title"] == "Docs"


def test_activitywatch_json_transition_edges_sort_unsorted_input(tmp_path):
    export = tmp_path / "aw.json"
    export.write_text(
        json.dumps(
            {
                "id": "window",
                "type": "currentwindow",
                "events": [
                    {
                        "timestamp": "2025-01-01T10:02:00Z",
                        "duration": 60,
                        "data": {"app": "Terminal", "title": "shell"},
                    },
                    {
                        "timestamp": "2025-01-01T10:00:00Z",
                        "duration": 60,
                        "data": {"app": "Code", "title": "main.py"},
                    },
                    {
                        "timestamp": "2025-01-01T10:01:00Z",
                        "duration": 60,
                        "data": {"app": "Firefox", "title": "Docs"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ActivityWatchJsonAdapter(path=str(export)).ingest()

    assert [edge.metadata["previous_app"] for edge in result.edges] == ["Code", "Firefox"]
    assert [edge.metadata["next_app"] for edge in result.edges] == ["Firefox", "Terminal"]


def test_activitywatch_json_transition_edges_do_not_cross_files(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "id": "window",
                "type": "currentwindow",
                "events": [
                    {"timestamp": "2025-01-01T10:00:00Z", "duration": 60, "data": {"app": "Code"}}
                ],
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "id": "window",
                "type": "currentwindow",
                "events": [
                    {
                        "timestamp": "2025-01-01T10:01:00Z",
                        "duration": 60,
                        "data": {"app": "Firefox"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ActivityWatchJsonAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert result.edges == []


def test_activitywatch_json_transition_edges_skip_cross_day_gap_but_allow_contiguous(tmp_path):
    export = tmp_path / "aw.json"
    export.write_text(
        json.dumps(
            {
                "id": "window",
                "type": "currentwindow",
                "events": [
                    {
                        "timestamp": "2025-01-01T23:59:00Z",
                        "duration": 30,
                        "data": {"app": "Code", "title": "gap"},
                    },
                    {
                        "timestamp": "2025-01-02T00:00:00Z",
                        "duration": 60,
                        "data": {"app": "Firefox", "title": "next"},
                    },
                    {
                        "timestamp": "2025-01-02T00:01:00Z",
                        "duration": 60,
                        "data": {"app": "Terminal", "title": "shell"},
                    },
                    {
                        "timestamp": "2025-01-02T23:59:00Z",
                        "duration": 60,
                        "data": {"app": "Code", "title": "contiguous"},
                    },
                    {
                        "timestamp": "2025-01-03T00:00:00Z",
                        "duration": 60,
                        "data": {"app": "Firefox", "title": "allowed"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ActivityWatchJsonAdapter(path=str(export)).ingest()

    assert [
        (edge.metadata["previous_title"], edge.metadata["next_title"]) for edge in result.edges
    ] == [
        ("next", "shell"),
        ("shell", "contiguous"),
        ("contiguous", "allowed"),
    ]


def test_activitywatch_json_transition_edges_respect_entity_types_filter(tmp_path):
    export = tmp_path / "aw.json"
    export.write_text(
        json.dumps(
            {
                "id": "window",
                "type": "currentwindow",
                "events": [
                    {"timestamp": "2025-01-01T10:00:00Z", "data": {"app": "Code"}},
                    {"timestamp": "2025-01-01T10:01:00Z", "data": {"app": "Firefox"}},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ActivityWatchJsonAdapter(path=str(export)).ingest(entity_types=["event"])

    assert result.units == []
    assert result.edges == []
