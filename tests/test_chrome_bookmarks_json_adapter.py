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

    result = ChromeBookmarksJsonAdapter(path=str(export)).ingest(entity_types=["bookmark", "folder"])

    assert sorted(unit.title for unit in result.units if unit.source_entity_type == "bookmark") == ["Graph Notes", "Other"]
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
    folder_edge = next(edge for edge in result.edges if edge.to_unit_id == unit.source_id)
    assert folder_edge.relation == "contains"
    assert folder_edge.metadata["folder_path"] == "Research"


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
    result = ChromeBookmarksJsonAdapter(path=str(export)).ingest(since=since, entity_types=["bookmark"])

    assert [unit.title for unit in result.units] == ["New"]
    assert ChromeBookmarksJsonAdapter(path=str(export)).ingest(entity_types=["history"]).units == []
    assert ChromeBookmarksJsonAdapter(path=str(export)).entity_types == ["bookmark", "domain", "folder"]


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


def test_chrome_bookmarks_json_emits_domain_aggregates_and_edges(tmp_path):
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
                                        "name": "A",
                                        "url": "https://Example.com/a",
                                        "guid": "guid-a",
                                        "date_added": "13253932800000000",
                                        "date_last_used": "13317004800000000",
                                    },
                                    {
                                        "type": "url",
                                        "name": "B",
                                        "url": "https://example.com/b",
                                        "guid": "guid-b",
                                        "date_added": "13285209600000000",
                                    },
                                ],
                            }
                        ],
                    },
                    "other": {
                        "type": "folder",
                        "children": [{"type": "url", "name": "C", "url": "https://other.example/c", "guid": "guid-c"}],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = ChromeBookmarksJsonAdapter(path=str(export)).ingest(entity_types=["bookmark", "domain"])
    domains = sorted((unit for unit in result.units if unit.source_entity_type == "domain"), key=lambda unit: unit.metadata["domain"])
    assert [unit.metadata["domain"] for unit in domains] == ["example.com", "other.example"]
    domain = domains[0]
    bookmarks = [unit for unit in result.units if unit.source_entity_type == "bookmark" and unit.metadata["domain"] == "example.com"]
    assert domain.metadata["bookmark_count"] == 2
    assert domain.metadata["roots"] == ["bookmark_bar"]
    assert domain.metadata["folder_paths"] == ["Research"]
    assert domain.metadata["bookmark_source_ids"] == sorted(unit.source_id for unit in bookmarks)
    assert domain.metadata["first_added_at"] == "2021-01-01T00:00:00+00:00"
    assert domain.metadata["last_used_at"] == "2023-01-01T00:00:00+00:00"
    assert {
        edge.to_unit_id
        for edge in result.edges
        if edge.from_unit_id == domain.source_id and edge.metadata.get("relation_type") == "domain_contains_bookmark"
    } == {
        unit.source_id for unit in bookmarks
    }

    domain_only = ChromeBookmarksJsonAdapter(path=str(export)).ingest(entity_types=["domain"])
    assert {unit.source_entity_type for unit in domain_only.units} == {"domain"}
    assert domain_only.edges == []


def test_chrome_bookmarks_json_emits_folder_units_and_filtered_edges(tmp_path):
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
                                    {"type": "url", "name": "A", "url": "https://example.com/a", "guid": "guid-a"},
                                    {"name": "Deep", "type": "folder", "children": [{"type": "url", "name": "B", "url": "https://example.com/b", "guid": "guid-b"}]},
                                ],
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = ChromeBookmarksJsonAdapter(path=str(export)).ingest(entity_types=["bookmark", "folder"])
    folders = {unit.metadata["folder_path"]: unit for unit in result.units if unit.source_entity_type == "folder"}

    assert set(folders) == {"", "Research", "Research/Deep"}
    assert folders["Research"].metadata["root"] == "bookmark_bar"
    assert folders["Research"].metadata["root_name"] == "Bookmarks Bar"
    assert folders["Research"].metadata["parent_folder_path"] == ""
    assert folders["Research"].metadata["child_bookmark_count"] == 1
    assert folders["Research"].metadata["child_folder_count"] == 1
    assert folders["Research"].metadata["bookmark_count"] == 2
    assert len(result.edges) == 2
    assert all(edge.from_unit_id in {folders["Research"].source_id, folders["Research/Deep"].source_id} for edge in result.edges)

    bookmark_only = ChromeBookmarksJsonAdapter(path=str(export)).ingest(entity_types=["bookmark"])
    folder_only = ChromeBookmarksJsonAdapter(path=str(export)).ingest(entity_types=["folder"])
    assert {unit.source_entity_type for unit in bookmark_only.units} == {"bookmark"}
    assert bookmark_only.edges == []
    assert {unit.source_entity_type for unit in folder_only.units} == {"folder"}
    assert folder_only.edges == []
