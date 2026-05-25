from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.mastodon_archive_outbox_json import MastodonArchiveOutboxJsonAdapter


def test_mastodon_archive_outbox_json_ingests_create_note(tmp_path):
    export = tmp_path / "outbox.json"
    export.write_text(
        json.dumps(
            {
                "orderedItems": [
                    {
                        "id": "https://example.social/users/me/statuses/1/activity",
                        "type": "Create",
                        "object": {
                            "id": "https://example.social/users/me/statuses/1",
                            "type": "Note",
                            "published": "2025-01-02T10:30:00Z",
                            "url": "https://example.social/@me/1",
                            "content": "<p>Hello <a href='https://example.com'>world</a> #Intro</p>",
                            "to": ["https://www.w3.org/ns/activitystreams#Public"],
                            "cc": ["https://example.social/users/me/followers"],
                            "tag": [{"type": "Hashtag", "name": "#Intro"}],
                            "attachment": [{"type": "Document", "url": "https://example.com/image.png"}],
                            "inReplyTo": "https://example.social/@you/0",
                        },
                    },
                    {"type": "Announce", "object": {}},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MastodonArchiveOutboxJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "mastodon_outbox_json"
    assert unit.source_entity_type == "note"
    assert unit.content == "Hello world #Intro"
    assert unit.metadata["url"] == "https://example.social/@me/1"
    assert unit.metadata["visibility"] == "public"
    assert unit.metadata["in_reply_to"] == "https://example.social/@you/0"
    assert unit.tags == ["intro"]
    assert unit.created_at == datetime(2025, 1, 2, 10, 30, tzinfo=timezone.utc)


def test_mastodon_archive_outbox_json_handles_top_level_list(tmp_path):
    export = tmp_path / "outbox.json"
    export.write_text(
        json.dumps([{"type": "Create", "object": {"type": "Note", "content": "<p>List item</p>"}}]),
        encoding="utf-8",
    )

    unit = MastodonArchiveOutboxJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.content == "List item"
    assert unit.source_id.startswith("mastodon_outbox_json:")
