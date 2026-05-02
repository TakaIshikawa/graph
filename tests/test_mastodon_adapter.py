"""Tests for the Mastodon ActivityPub outbox adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.mastodon import MastodonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


PUBLIC = "https://www.w3.org/ns/activitystreams#Public"


def test_get_adapter_returns_mastodon_instance(tmp_path):
    export = tmp_path / "outbox.json"
    adapter = get_adapter("mastodon", path=str(export))

    assert isinstance(adapter, MastodonAdapter)
    assert adapter.name == "mastodon"


def test_ingest_outbox_create_notes_as_artifacts(tmp_path):
    export = tmp_path / "outbox.json"
    export.write_text(
        json.dumps(
            {
                "type": "OrderedCollection",
                "orderedItems": [
                    {
                        "type": "Create",
                        "object": {
                            "id": "https://example.social/users/alice/statuses/1",
                            "type": "Note",
                            "url": "https://example.social/@alice/1",
                            "attributedTo": "https://example.social/users/alice",
                            "content": (
                                "<p>Hello <strong>Graph</strong><br />"
                                "Testing #PKM &amp; #Graph-Notes</p>"
                            ),
                            "published": "2024-01-02T03:04:05Z",
                            "updated": "2024-01-03T04:05:06+00:00",
                            "conversation": "tag:example.social,2024-01-02:objectId=1",
                            "sensitive": True,
                            "to": [PUBLIC],
                            "cc": ["https://example.social/users/alice/followers"],
                            "tag": [
                                {
                                    "type": "Hashtag",
                                    "name": "#Graph",
                                    "href": "https://example.social/tags/graph",
                                },
                                {"type": "Mention", "name": "@bob@example.social"},
                            ],
                        },
                    },
                    {
                        "type": "Announce",
                        "object": {"type": "Note", "content": "Boosts are skipped"},
                    },
                    {
                        "type": "Create",
                        "object": {"type": "Article", "content": "Articles are skipped"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = MastodonAdapter(path=str(export)).ingest()

    assert result.edges == []
    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.MASTODON
    assert unit.source_id == "https://example.social/users/alice/statuses/1"
    assert unit.source_entity_type == "note"
    assert unit.title == "Hello Graph Testing #PKM & #Graph-Notes"
    assert unit.content == "Hello Graph\nTesting #PKM & #Graph-Notes"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.tags == ["graph", "graph-notes", "pkm"]
    assert unit.created_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2024, 1, 3, 4, 5, 6, tzinfo=timezone.utc)
    assert unit.metadata == {
        "url": "https://example.social/@alice/1",
        "attributedTo": "https://example.social/users/alice",
        "conversation": "tag:example.social,2024-01-02:objectId=1",
        "sensitive": True,
        "visibility": "public",
        "to": [PUBLIC],
        "cc": ["https://example.social/users/alice/followers"],
        "published": "2024-01-02T03:04:05Z",
        "updated": "2024-01-03T04:05:06+00:00",
        "source_file": "outbox.json",
    }


def test_ingest_directory_path_finds_outbox_json_and_uses_url_source_id(tmp_path):
    export = tmp_path / "outbox.json"
    export.write_text(
        json.dumps(
            {
                "orderedItems": [
                    {
                        "type": "Create",
                        "object": {
                            "type": "Note",
                            "url": "https://example.social/@alice/2",
                            "content": "<p>URL fallback #Archive</p>",
                            "published": "2024-02-01T00:00:00+09:00",
                            "to": ["https://example.social/users/alice/followers"],
                            "cc": [],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MastodonAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "https://example.social/@alice/2"
    assert unit.tags == ["archive"]
    assert unit.metadata["visibility"] == "private"
    assert unit.created_at == datetime(2024, 1, 31, 15, 0, 0, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2024, 1, 31, 15, 0, 0, tzinfo=timezone.utc)


def test_since_filter_uses_updated_timestamp(tmp_path):
    export = tmp_path / "outbox.json"
    export.write_text(
        json.dumps(
            {
                "orderedItems": [
                    {
                        "type": "Create",
                        "object": {
                            "id": "old",
                            "type": "Note",
                            "content": "Old",
                            "published": "2024-01-01T00:00:00Z",
                            "updated": "2024-01-02T00:00:00Z",
                        },
                    },
                    {
                        "type": "Create",
                        "object": {
                            "id": "new",
                            "type": "Note",
                            "content": "New",
                            "published": "2024-01-01T00:00:00Z",
                            "updated": "2024-01-04T00:00:00Z",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MastodonAdapter(path=str(export)).ingest(
        since=SyncState(
            source_project="mastodon",
            source_entity_type="note",
            last_sync_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        )
    )

    assert [unit.source_id for unit in result.units] == ["new"]


def test_entity_types_filtering_returns_no_notes_for_other_types(tmp_path):
    export = tmp_path / "outbox.json"
    export.write_text(
        json.dumps(
            {
                "orderedItems": [
                    {
                        "type": "Create",
                        "object": {"id": "1", "type": "Note", "content": "Filtered"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MastodonAdapter(path=str(export)).ingest(entity_types=["status"])

    assert result.units == []
    assert result.edges == []


def test_missing_and_malformed_files_return_empty_result(tmp_path):
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "outbox.json"
    malformed.write_text("{not json", encoding="utf-8")

    assert MastodonAdapter(path=str(missing)).ingest().units == []
    assert MastodonAdapter(path=str(malformed)).ingest().units == []
