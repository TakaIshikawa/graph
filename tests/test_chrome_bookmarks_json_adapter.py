from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.chrome_bookmarks_json import ChromeBookmarksJsonAdapter
from graph.types.models import SyncState


def test_chrome_bookmarks_json_ingests_nested_bookmarks_dates_and_edges(tmp_path):
    export = tmp_path / "Bookmarks.json"
    export.write_text(
        json.dumps(
            {
                "roots": {
                    "bookmark_bar": {
                        "name": "Bookmarks Bar",
                        "type": "folder",
                        "children": [
                            {
                                "name": "Research",
                                "type": "folder",
                                "children": [
                                    {
                                        "type": "url",
                                        "name": "Graph Notes",
                                        "url": "https://example.com/graph",
                                        "guid": "guid-1",
                                        "id": "10",
                                        "date_added": "13253932800000000",
                                        "date_last_used": "13317004800000000",
                                    }
                                ],
                            }
                        ],
                    },
                    "other": {
                        "name": "Other Bookmarks",
                        "type": "folder",
                        "children": [
                            {"type": "url", "name": "Other", "url": "https://other.example", "guid": "guid-2"}
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = ChromeBookmarksJsonAdapter(path=str(export)).ingest()

    assert sorted(unit.title for unit in result.units) == ["Graph Notes", "Other"]
    unit = next(unit for unit in result.units if unit.title == "Graph Notes")
    assert unit.source_project == "chrome_bookmarks_json"
    assert unit.source_entity_type == "bookmark"
    assert unit.metadata["url"] == "https://example.com/graph"
    assert unit.metadata["domain"] == "example.com"
    assert unit.metadata["folder_path"] == "Research"
    assert unit.metadata["root"] == "bookmark_bar"
    assert unit.metadata["root_name"] == "Bookmarks Bar"
    assert unit.metadata["date_added"] == "2021-01-01T00:00:00+00:00"
    assert unit.metadata["date_last_used"] == "2023-01-01T00:00:00+00:00"
    assert unit.content == "Graph Notes\nURL: https://example.com/graph\nRoot: Bookmarks Bar\nFolder: Research"
    assert result.edges[0].to_unit_id == unit.source_id
    assert result.edges[0].relation == "contains"
    assert result.edges[0].metadata["folder_path"] == "Research"


def test_chrome_bookmarks_json_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "Bookmarks.json"
    export.write_text(
        json.dumps(
            {
                "roots": {
                    "bookmark_bar": {
                        "type": "folder",
                        "children": [
                            {"type": "url", "name": "Old", "url": "https://old.example", "date_added": "13253760000000000"},
                            {"type": "url", "name": "New", "url": "https://new.example", "date_added": "13317004800000000"},
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    since = SyncState(source_project="chrome_bookmarks_json", source_entity_type="bookmark", last_sync_at=datetime(2022, 1, 1, tzinfo=timezone.utc))
    result = ChromeBookmarksJsonAdapter(path=str(export)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["New"]
    assert ChromeBookmarksJsonAdapter(path=str(export)).ingest(entity_types=["history"]).units == []


def test_chrome_bookmarks_json_source_id_fallback_is_stable(tmp_path):
    export = tmp_path / "Bookmarks.json"
    export.write_text(
        json.dumps({"roots": {"synced": {"type": "folder", "children": [{"type": "url", "name": "Mobile", "url": "https://mobile.example"}]}}}),
        encoding="utf-8",
    )

    first = ChromeBookmarksJsonAdapter(path=str(export)).ingest().units[0]
    second = ChromeBookmarksJsonAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.metadata["root_name"] == "Mobile Bookmarks"
