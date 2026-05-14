from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.discord_json import DiscordJsonAdapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def test_discord_json_adapter_ingests_messages_metadata_and_reply_edges(tmp_path):
    export_path = tmp_path / "general.json"
    export_path.write_text(
        json.dumps(
            {
                "guild": {"id": "guild-1", "name": "Research Lab"},
                "channel": {"id": "channel-1", "name": "general"},
                "messages": [
                    {
                        "id": "100",
                        "timestamp": "2026-04-01T10:00:00.000+00:00",
                        "content": "The first note has a useful link.",
                        "author": {
                            "id": "user-1",
                            "username": "alice",
                            "globalName": "Alice A.",
                        },
                        "attachments": [
                            {
                                "id": "att-1",
                                "filename": "diagram.png",
                                "url": "https://cdn.discordapp.com/diagram.png",
                                "content_type": "image/png",
                                "size": 1234,
                            }
                        ],
                    },
                    {
                        "id": "101",
                        "timestamp": "2026-04-01T10:05:00Z",
                        "editedTimestamp": "2026-04-01T10:06:00Z",
                        "content": "Replying with follow-up context.",
                        "author": {"id": "user-2", "username": "bob"},
                        "messageReference": {
                            "messageId": "100",
                            "channelId": "channel-1",
                            "guildId": "guild-1",
                            "timestamp": "2026-04-01T10:00:00.000+00:00",
                            "author": {
                                "id": "user-1",
                                "username": "alice",
                                "globalName": "Alice A.",
                            },
                        },
                    },
                    {
                        "id": "102",
                        "timestamp": "2026-04-01T10:07:00Z",
                        "content": "   ",
                        "author": {"id": "user-2", "username": "bob"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = DiscordJsonAdapter(path=str(export_path)).ingest(entity_types=["discord_message"])

    assert [unit.source_id for unit in result.units] == [
        "discord_json:channel-1:100",
        "discord_json:channel-1:101",
    ]
    first, second = result.units
    assert first.source_project == SourceProject.DISCORD_JSON
    assert first.source_entity_type == "discord_message"
    assert first.title == "#general 2026-04-01 Alice A."
    assert first.content == "The first note has a useful link."
    assert first.metadata["server_name"] == "Research Lab"
    assert first.metadata["channel_name"] == "general"
    assert first.metadata["author"]["id"] == "user-1"
    assert first.metadata["attachments"] == [
        {
            "id": "att-1",
            "filename": "diagram.png",
            "url": "https://cdn.discordapp.com/diagram.png",
            "content_type": "image/png",
            "size": 1234,
        }
    ]
    assert first.tags == [
        "discord",
        "discord-server-research-lab",
        "discord-channel-general",
    ]
    assert first.created_at == datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    assert second.updated_at == datetime(2026, 4, 1, 10, 6, tzinfo=timezone.utc)
    assert second.metadata["references"] == [
        {
            "message_id": "100",
            "channel_id": "channel-1",
            "server_id": "guild-1",
            "timestamp": "2026-04-01T10:00:00.000+00:00",
            "author": {
                "id": "user-1",
                "username": "alice",
                "display_name": "Alice A.",
                "discriminator": "",
                "bot": "",
            },
        }
    ]

    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_unit_id == "discord_json:channel-1:101"
    assert edge.to_unit_id == "discord_json:channel-1:100"
    assert edge.relation == EdgeRelation.REFERENCES
    assert edge.source == EdgeSource.SOURCE
    assert edge.metadata["relation_type"] == "discord_reply_reference"
    assert edge.metadata["referenced_message_id"] == "100"
    assert edge.metadata["referenced_channel_id"] == "channel-1"
    assert edge.metadata["referenced_server_id"] == "guild-1"
    assert edge.metadata["referenced_timestamp"] == "2026-04-01T10:00:00.000+00:00"
    assert edge.metadata["referenced_author"]["id"] == "user-1"


def test_discord_json_adapter_reads_directory_and_ignores_malformed_optional_fields(tmp_path):
    channel_dir = tmp_path / "messages" / "channel-2"
    channel_dir.mkdir(parents=True)
    (channel_dir / "channel.json").write_text(
        json.dumps({"id": "channel-2", "name": "ideas"}),
        encoding="utf-8",
    )
    (channel_dir / "messages.json").write_text(
        json.dumps(
            [
                {
                    "ID": "200",
                    "Timestamp": "2026-04-02T09:00:00Z",
                    "Contents": "Attachment metadata should survive.",
                    "Author": "carol",
                    "Attachments": "not-a-list",
                    "message_reference": 123,
                },
                {
                    "ID": "201",
                    "Timestamp": "2026-04-02T09:01:00Z",
                    "Contents": "",
                    "Attachments": [
                        "https://cdn.discordapp.com/file.txt",
                        {"fileName": "notes.txt", "url": "https://cdn.discordapp.com/notes.txt"},
                        {"unexpected": ["nested"]},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = DiscordJsonAdapter(root_path=str(tmp_path)).ingest(entity_types=["discord_message"])

    assert [unit.source_id for unit in result.units] == [
        "discord_json:channel-2:200",
        "discord_json:channel-2:201",
    ]
    assert result.units[0].metadata["attachments"] == []
    assert "references" not in result.units[0].metadata
    assert result.units[1].content == "https://cdn.discordapp.com/file.txt notes.txt"
    assert result.units[1].metadata["attachments"] == [
        {"url": "https://cdn.discordapp.com/file.txt"},
        {"filename": "notes.txt", "url": "https://cdn.discordapp.com/notes.txt"},
        {},
    ]
    assert result.edges == []


def test_discord_json_adapter_since_and_entity_type_filters(tmp_path):
    export_path = tmp_path / "messages.json"
    export_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "id": "300",
                        "timestamp": "2026-04-01T10:00:00Z",
                        "content": "Old message.",
                    },
                    {
                        "id": "301",
                        "timestamp": "2026-04-03T10:00:00Z",
                        "content": "New message.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    adapter = DiscordJsonAdapter(path=str(export_path))

    filtered = adapter.ingest(
        since=SyncState(
            source_project="discord_json",
            source_entity_type="discord_message",
            last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
        )
    )
    excluded = adapter.ingest(entity_types=["other"])

    assert [unit.metadata["message_id"] for unit in filtered.units] == ["301"]
    assert excluded.units == []
    assert excluded.edges == []


def test_discord_json_adapter_emits_attachment_units_and_source_edges(tmp_path):
    export_path = tmp_path / "general.json"
    export_path.write_text(
        json.dumps(
            {
                "guild": {"id": "guild-1", "name": "Research Lab"},
                "channel": {"id": "channel-1", "name": "general"},
                "messages": [
                    {
                        "id": "400",
                        "timestamp": "2026-04-04T10:00:00Z",
                        "content": "See attached evidence.",
                        "author": {"id": "user-1", "username": "alice", "globalName": "Alice A."},
                        "attachments": [
                            {
                                "id": "att-1",
                                "filename": "diagram.png",
                                "url": "https://cdn.discordapp.com/diagram.png",
                                "content_type": "image/png",
                                "size": 1234,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = DiscordJsonAdapter(path=str(export_path)).ingest()

    assert DiscordJsonAdapter(path=str(export_path)).entity_types == ["discord_message", "discord_attachment", "discord_channel"]
    assert [unit.source_entity_type for unit in result.units] == ["discord_message", "discord_attachment"]
    attachment = result.units[1]
    assert attachment.source_id == "discord_json:channel-1:400:attachment:att-1"
    assert attachment.title == "diagram.png"
    assert attachment.content == "diagram.png https://cdn.discordapp.com/diagram.png image/png"
    assert attachment.metadata["filename"] == "diagram.png"
    assert attachment.metadata["url"] == "https://cdn.discordapp.com/diagram.png"
    assert attachment.metadata["content_type"] == "image/png"
    assert attachment.metadata["size"] == 1234
    assert attachment.metadata["message_id"] == "400"
    assert attachment.metadata["channel_name"] == "general"
    assert attachment.metadata["server_name"] == "Research Lab"
    assert attachment.metadata["author"]["display_name"] == "Alice A."
    assert attachment.metadata["timestamp"] == "2026-04-04T10:00:00Z"
    assert attachment.metadata["source_path"] == "general.json"

    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_unit_id == "discord_json:channel-1:400"
    assert edge.to_unit_id == attachment.source_id
    assert edge.relation == EdgeRelation.CONTAINS
    assert edge.source == EdgeSource.SOURCE
    assert edge.metadata["relation_type"] == "discord_message_attachment"
    assert edge.metadata["attachment_id"] == "att-1"


def test_discord_json_adapter_attachment_only_messages_are_searchable(tmp_path):
    export_path = tmp_path / "attachments.json"
    export_path.write_text(
        json.dumps(
            {
                "channel": {"id": "channel-2", "name": "media"},
                "messages": [
                    {
                        "id": "401",
                        "timestamp": "2026-04-04T11:00:00Z",
                        "content": "",
                        "attachments": [{"filename": "notes.txt", "url": "https://cdn.discordapp.com/notes.txt"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = DiscordJsonAdapter(path=str(export_path)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["discord_message", "discord_attachment"]
    assert result.units[0].content == "notes.txt"
    attachment = result.units[1]
    assert attachment.title == "notes.txt"
    assert attachment.content == "notes.txt https://cdn.discordapp.com/notes.txt"
    assert attachment.source_id.startswith("discord_json:channel-2:401:attachment:")
    assert result.edges[0].metadata["relation_type"] == "discord_message_attachment"


def test_discord_json_adapter_entity_type_filtering_for_attachments(tmp_path):
    export_path = tmp_path / "messages.json"
    export_path.write_text(
        json.dumps(
            {
                "channel": {"id": "channel-3", "name": "files"},
                "messages": [
                    {
                        "id": "402",
                        "timestamp": "2026-04-04T12:00:00Z",
                        "content": "Message with an attachment.",
                        "attachments": [{"id": "att-2", "filename": "report.pdf"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    adapter = DiscordJsonAdapter(path=str(export_path))

    messages = adapter.ingest(entity_types=["discord_message"])
    attachments = adapter.ingest(entity_types=["discord_attachment"])

    assert [unit.source_entity_type for unit in messages.units] == ["discord_message"]
    assert messages.edges == []
    assert [unit.source_entity_type for unit in attachments.units] == ["discord_attachment"]
    assert attachments.units[0].source_id == "discord_json:channel-3:402:attachment:att-2"
    assert attachments.edges == []


def test_discord_json_adapter_skips_malformed_attachment_units(tmp_path):
    export_path = tmp_path / "messages.json"
    export_path.write_text(
        json.dumps(
            {
                "channel": {"id": "channel-4", "name": "files"},
                "messages": [
                    {
                        "id": "403",
                        "timestamp": "2026-04-04T13:00:00Z",
                        "content": "Malformed attachments should not crash.",
                        "attachments": [{"unexpected": ["nested"]}, 123, None],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = DiscordJsonAdapter(path=str(export_path)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["discord_message"]
    assert result.units[0].metadata["attachments"] == [{}]
    assert result.edges == []


def test_discord_json_adapter_emits_channel_aggregate_and_edges(tmp_path):
    export_path = tmp_path / "general.json"
    export_path.write_text(
        json.dumps(
            {
                "guild": {"id": "guild-1", "name": "Research Lab"},
                "channel": {"id": "channel-1", "name": "general"},
                "messages": [
                    {
                        "id": "500",
                        "timestamp": "2026-04-04T10:00:00Z",
                        "content": "First.",
                        "author": {"id": "user-1", "username": "alice"},
                        "attachments": [{"id": "att-1", "filename": "diagram.png"}],
                    },
                    {
                        "id": "501",
                        "timestamp": "2026-04-04T10:05:00Z",
                        "content": "Second.",
                        "author": {"id": "user-2", "username": "bob"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = DiscordJsonAdapter(path=str(export_path)).ingest(entity_types=["discord_message", "discord_channel"])

    channel = next(unit for unit in result.units if unit.source_entity_type == "discord_channel")
    messages = [unit for unit in result.units if unit.source_entity_type == "discord_message"]
    assert channel.source_id == "discord_json:channel:channel-1"
    assert channel.metadata["channel_id"] == "channel-1"
    assert channel.metadata["channel_name"] == "general"
    assert channel.metadata["message_count"] == 2
    assert channel.metadata["attachment_count"] == 1
    assert channel.metadata["author_count"] == 2
    assert channel.metadata["server_ids"] == ["guild-1"]
    assert channel.metadata["server_names"] == ["Research Lab"]
    assert channel.metadata["first_message_at"] == "2026-04-04T10:00:00+00:00"
    assert channel.metadata["last_message_at"] == "2026-04-04T10:05:00+00:00"
    assert channel.metadata["source_paths"] == ["general.json"]

    channel_edges = [edge for edge in result.edges if edge.metadata.get("relation_type") == "discord_channel_message"]
    assert len(channel_edges) == 2
    assert {edge.from_unit_id for edge in channel_edges} == {channel.source_id}
    assert {edge.to_unit_id for edge in channel_edges} == {unit.source_id for unit in messages}
    assert {edge.relation for edge in channel_edges} == {EdgeRelation.CONTAINS}


def test_discord_json_adapter_channel_entity_filtering(tmp_path):
    export_path = tmp_path / "general.json"
    export_path.write_text(
        json.dumps({"channel": {"id": "channel-1", "name": "general"}, "messages": [{"id": "500", "content": "First."}]}),
        encoding="utf-8",
    )

    result = DiscordJsonAdapter(path=str(export_path)).ingest(entity_types=["discord_channel"])

    assert [unit.source_entity_type for unit in result.units] == ["discord_channel"]
    assert result.edges == []
