"""Tests for the Safari Bookmarks.plist adapter."""

from __future__ import annotations

import plistlib
from datetime import datetime, timezone

import pytest

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.safari_bookmarks import SafariBookmarksAdapter
from graph.types.models import SyncState


def _write_plist(path, value) -> None:
    with path.open("wb") as handle:
        plistlib.dump(value, handle)


def test_get_adapter_returns_safari_bookmarks_instance(tmp_path):
    export = tmp_path / "Bookmarks.plist"
    adapter = get_adapter("safari_bookmarks", path=str(export))

    assert isinstance(adapter, SafariBookmarksAdapter)
    assert adapter.name == "safari_bookmarks"
    assert "safari_bookmarks" in list_adapters()


def test_ingest_nested_safari_bookmarks_preserves_metadata(tmp_path):
    export = tmp_path / "Bookmarks.plist"
    created_at = datetime(2024, 1, 2, 3, 4, 5)
    updated_at = datetime(2024, 1, 3, 4, 5, 6, tzinfo=timezone.utc)
    _write_plist(
        export,
        {
            "Children": [
                {
                    "Title": "Favorites",
                    "Children": [
                        {
                            "Title": "Research",
                            "Children": [
                                {
                                    "WebBookmarkType": "WebBookmarkTypeLeaf",
                                    "URLString": "https://example.com/graph",
                                    "URIDictionary": {"title": "Graph Notes"},
                                    "WebBookmarkUUID": "bookmark-uuid",
                                    "DateAdded": created_at,
                                    "DateLastModified": updated_at,
                                }
                            ],
                        }
                    ],
                },
                {
                    "Title": "Reading List",
                    "Children": [
                        {
                            "URLString": "https://example.com/reading",
                            "Title": "Reading Item",
                            "WebBookmarkIdentifier": "reading-id",
                        }
                    ],
                },
            ]
        },
    )

    result = SafariBookmarksAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "safari_bookmarks:bookmark-uuid",
        "safari_bookmarks:reading-id",
    ]
    unit = result.units[0]
    assert unit.source_project == "safari_bookmarks"
    assert unit.source_entity_type == "bookmark"
    assert unit.title == "Graph Notes"
    assert unit.content == "Graph Notes\nURL: https://example.com/graph\nFolder: Favorites/Research"
    assert unit.content_type == "artifact"
    assert unit.metadata == {
        "url": "https://example.com/graph",
        "folder_path": "Favorites/Research",
        "title": "Graph Notes",
        "uuid": "bookmark-uuid",
        "safari_id": "bookmark-uuid",
        "created_at": "2024-01-02T03:04:05+00:00",
        "updated_at": "2024-01-03T04:05:06+00:00",
        "source_file": "Bookmarks.plist",
    }
    assert unit.tags == ["Favorites", "Favorites/Research"]
    assert unit.created_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2024, 1, 3, 4, 5, 6, tzinfo=timezone.utc)


def test_missing_empty_filtered_and_non_bookmark_plists_return_empty_result(tmp_path):
    missing = tmp_path / "missing.plist"
    empty = tmp_path / "empty.plist"
    non_bookmark = tmp_path / "non_bookmark.plist"
    valid = tmp_path / "valid.plist"
    _write_plist(empty, {})
    _write_plist(non_bookmark, ["not a safari plist"])
    _write_plist(
        valid,
        {
            "Children": [
                {
                    "URLString": "https://example.com",
                    "URIDictionary": {"title": "Filtered"},
                }
            ]
        },
    )

    assert SafariBookmarksAdapter(path=str(missing)).ingest().units == []
    assert SafariBookmarksAdapter(path=str(empty)).ingest().units == []
    assert SafariBookmarksAdapter(path=str(non_bookmark)).ingest().units == []
    assert SafariBookmarksAdapter(path=str(valid)).ingest(entity_types=["article"]).units == []


def test_malformed_plist_raises_clear_value_error(tmp_path):
    malformed = tmp_path / "Bookmarks.plist"
    malformed.write_bytes(b"not a plist")

    with pytest.raises(ValueError, match="Could not read Safari bookmarks plist"):
        SafariBookmarksAdapter(path=str(malformed)).ingest()


def test_url_source_id_fallback_is_stable(tmp_path):
    export = tmp_path / "Bookmarks.plist"
    _write_plist(
        export,
        {
            "Children": [
                {
                    "Title": "Favorites",
                    "Children": [
                        {
                            "URLString": "https://example.com/stable",
                            "URIDictionary": {"title": "Stable"},
                        }
                    ],
                }
            ]
        },
    )

    first = SafariBookmarksAdapter(path=str(export)).ingest().units[0]
    second = SafariBookmarksAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("url:")


def test_since_filter_uses_updated_or_created_datetime(tmp_path):
    export = tmp_path / "Bookmarks.plist"
    _write_plist(
        export,
        {
            "Children": [
                {
                    "URLString": "https://example.com/old",
                    "URIDictionary": {"title": "Old"},
                    "WebBookmarkUUID": "old",
                    "DateAdded": datetime(2024, 1, 1, tzinfo=timezone.utc),
                },
                {
                    "URLString": "https://example.com/new",
                    "URIDictionary": {"title": "New"},
                    "WebBookmarkUUID": "new",
                    "DateAdded": datetime(2024, 1, 3, tzinfo=timezone.utc),
                },
            ]
        },
    )

    result = SafariBookmarksAdapter(path=str(export)).ingest(
        since=SyncState(
            source_project="safari_bookmarks",
            source_entity_type="bookmark",
            last_sync_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    )

    assert [unit.source_id for unit in result.units] == ["safari_bookmarks:new"]
