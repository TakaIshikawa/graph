from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.discord_saved_messages_json import DiscordSavedMessagesJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_discord_saved_messages_json_ingests_plain_attachments_embeds_and_replies(tmp_path):
    export = tmp_path / "messages.json"
    export.write_text(
        json.dumps(
            {
                "guild": {"name": "Guild"},
                "channel": {"name": "links"},
                "messages": [
                    {
                        "id": "m1",
                        "author": {"username": "ada"},
                        "content": "Useful note",
                        "timestamp": "2025-01-01T00:00:00Z",
                        "attachments": [{"filename": "diagram.png", "url": "https://cdn.example/diagram.png"}],
                        "embeds": [{"title": "Embed title", "url": "https://example.com"}],
                        "message_reference": {"message_id": "m0", "channel_id": "c1"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = DiscordSavedMessagesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.DISCORD_SAVED_MESSAGES_JSON
    assert unit.source_id == "discord_saved_messages_json:m1"
    assert unit.metadata["guild"] == "Guild"
    assert unit.metadata["channel"] == "links"
    assert unit.metadata["author"] == "ada"
    assert unit.metadata["attachments"] == [{"filename": "diagram.png", "url": "https://cdn.example/diagram.png"}]
    assert unit.metadata["embeds"] == [{"title": "Embed title", "url": "https://example.com"}]
    assert unit.metadata["reply_to"] == {"message_id": "m0", "channel_id": "c1"}
    assert unit.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert get_adapter("discord_saved_messages_json", path=str(export)).name == "discord_saved_messages_json"


def test_discord_saved_messages_json_keeps_empty_content_with_media_and_filters(tmp_path):
    export = tmp_path / "messages.json"
    export.write_text(
        json.dumps(
            [
                {"id": "m1", "content": "", "attachments": [{"filename": "file.txt"}], "timestamp": "2025-01-02T00:00:00Z"},
                {"id": "m2", "content": ""},
            ]
        ),
        encoding="utf-8",
    )

    result = DiscordSavedMessagesJsonAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in result.units] == ["discord_saved_messages_json:m1"]
    assert result.units[0].metadata["attachments"] == [{"filename": "file.txt"}]
    assert DiscordSavedMessagesJsonAdapter(path=str(export)).ingest(entity_types=["channel"]).units == []
