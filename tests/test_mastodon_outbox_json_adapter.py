from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.mastodon_outbox_json import MastodonOutboxJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


PUBLIC = "https://www.w3.org/ns/activitystreams#Public"


def test_mastodon_outbox_json_ingests_create_note_records(tmp_path):
    export = tmp_path / "outbox.json"
    export.write_text(
        json.dumps(
            {
                "type": "OrderedCollection",
                "orderedItems": [
                    {
                        "id": "https://example.social/users/alice/statuses/1/activity",
                        "type": "Create",
                        "published": "2024-01-02T03:04:05Z",
                        "object": {
                            "id": "https://example.social/users/alice/statuses/1",
                            "type": "Note",
                            "url": "https://example.social/@alice/1",
                            "attributedTo": "https://example.social/users/alice",
                            "content": (
                                "<p>Hello <strong>Graph</strong><br />"
                                'See <a href="https://example.com">https://example.com</a> '
                                "#PKM &amp; #Graph-Notes</p>"
                            ),
                            "published": "2024-01-02T03:04:05Z",
                            "to": [PUBLIC],
                            "cc": ["https://example.social/users/alice/followers"],
                            "tag": [
                                {
                                    "type": "Hashtag",
                                    "name": "#Graph",
                                    "href": "https://example.social/tags/graph",
                                },
                                {
                                    "type": "Mention",
                                    "name": "@bob@example.social",
                                    "href": "https://example.social/@bob",
                                },
                            ],
                            "inReplyTo": "https://example.social/users/bob/statuses/9",
                            "replies": {"type": "Collection", "totalItems": 2},
                            "shares": {"type": "Collection", "totalItems": 1},
                        },
                    },
                    {
                        "type": "Announce",
                        "object": "https://example.social/users/charlie/statuses/3",
                    },
                    {
                        "type": "Create",
                        "object": {"type": "Article", "content": "Skipped"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = MastodonOutboxJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.MASTODON_OUTBOX_JSON
    assert unit.source_id == "https://example.social/users/alice/statuses/1"
    assert unit.source_entity_type == "note"
    assert unit.content == "Hello Graph\nSee https://example.com #PKM & #Graph-Notes"
    assert unit.tags == ["graph", "graph-notes", "pkm"]
    assert unit.created_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.metadata["activity_id"] == "https://example.social/users/alice/statuses/1/activity"
    assert unit.metadata["note_id"] == "https://example.social/users/alice/statuses/1"
    assert unit.metadata["url"] == "https://example.social/@alice/1"
    assert unit.metadata["published"] == "2024-01-02T03:04:05Z"
    assert unit.metadata["visibility"] == "public"
    assert unit.metadata["to"] == [PUBLIC]
    assert unit.metadata["cc"] == ["https://example.social/users/alice/followers"]
    assert unit.metadata["tags"] == ["graph", "graph-notes", "pkm"]
    assert unit.metadata["tag"][1]["href"] == "https://example.social/@bob"
    assert unit.metadata["in_reply_to"] == "https://example.social/users/bob/statuses/9"
    assert unit.metadata["shares"]["totalItems"] == 1
    assert unit.metadata["source_file"] == "outbox.json"
    assert unit.metadata["record_index"] == 0


def test_mastodon_outbox_json_supports_directory_registry_filters_and_bad_files(tmp_path):
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
                            "content": "<p>Private note</p>",
                            "published": "2024-02-01T00:00:00+09:00",
                            "updated": "2024-02-02T00:00:00+09:00",
                            "to": ["https://example.social/users/alice/followers"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    adapter = get_adapter("mastodon_outbox_json", path=str(tmp_path))
    result = adapter.ingest(
        since=SyncState(
            source_project="mastodon_outbox_json",
            source_entity_type="note",
            last_sync_at=datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc),
        )
    )

    assert isinstance(adapter, MastodonOutboxJsonAdapter)
    assert adapter.name == "mastodon_outbox_json"
    assert len(result.units) == 1
    assert result.units[0].metadata["visibility"] == "private"
    assert result.units[0].updated_at == datetime(2024, 2, 1, 15, 0, tzinfo=timezone.utc)
    assert adapter.ingest(entity_types=["status"]).units == []
    assert MastodonOutboxJsonAdapter(path=str(tmp_path / "missing.json")).ingest().units == []
