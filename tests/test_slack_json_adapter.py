from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.slack_json import SlackJsonAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_slack_json_ingests_single_file_messages(tmp_path):
    export = tmp_path / "general.json"
    _write_json(
        export,
        [
            {
                "type": "message",
                "user": "U123",
                "text": "Read <https://example.com/report|the report> and https://docs.example.com/.",
                "ts": "1712345678.000100",
                "client_msg_id": "client-1",
            }
        ],
    )

    result = SlackJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.SLACK_JSON
    assert unit.source_id == "slack_json:general:1712345678.000100"
    assert unit.source_entity_type == "slack_message"
    assert unit.title == "#general 2024-04-05 U123"
    assert unit.content == "Read the report and https://docs.example.com/."
    assert unit.content_type == ContentType.INSIGHT
    assert unit.metadata["channel"] == "general"
    assert unit.metadata["user"] == "U123"
    assert unit.metadata["ts"] == "1712345678.000100"
    assert unit.metadata["client_msg_id"] == "client-1"
    assert unit.metadata["links"] == [
        "https://example.com/report",
        "https://docs.example.com/",
    ]
    assert unit.created_at == datetime.fromtimestamp(1712345678.0001, tz=timezone.utc)
    assert result.edges == []


def test_slack_json_ingests_channel_directory_in_file_order(tmp_path):
    channel = tmp_path / "research"
    channel.mkdir()
    _write_json(
        channel / "2024-04-05.json",
        [
            {"type": "message", "user": "U1", "text": "First", "ts": "1712300000.000001"}
        ],
    )
    _write_json(
        channel / "2024-04-06.json",
        [
            {"type": "message", "user": "U2", "text": "Second", "ts": "1712400000.000002"}
        ],
    )

    result = SlackJsonAdapter(path=str(channel)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "slack_json:research:1712300000.000001",
        "slack_json:research:1712400000.000002",
    ]
    assert [unit.metadata["source_path"] for unit in result.units] == [
        "2024-04-05.json",
        "2024-04-06.json",
    ]
    assert [unit.title for unit in result.units] == [
        "#research 2024-04-05 U1",
        "#research 2024-04-06 U2",
    ]


def test_slack_json_preserves_thread_metadata(tmp_path):
    channel = tmp_path / "design"
    channel.mkdir()
    _write_json(
        channel / "2024-04-05.json",
        [
            {
                "type": "message",
                "user": "U1",
                "text": "Root message",
                "ts": "1712345000.000100",
                "thread_ts": "1712345000.000100",
                "reply_count": 1,
            },
            {
                "type": "message",
                "user": "U2",
                "text": "Reply message",
                "ts": "1712345100.000200",
                "thread_ts": "1712345000.000100",
            },
        ],
    )

    result = SlackJsonAdapter(path=str(channel)).ingest()

    root, reply = result.units
    assert root.metadata["thread_ts"] == "1712345000.000100"
    assert root.metadata["is_thread_reply"] is False
    assert root.metadata["reply_count"] == 1
    assert reply.metadata["thread_ts"] == "1712345000.000100"
    assert reply.metadata["is_thread_reply"] is True


def test_slack_json_skips_deleted_and_empty_messages(tmp_path):
    export = tmp_path / "random.json"
    _write_json(
        export,
        [
            {
                "type": "message",
                "subtype": "message_deleted",
                "text": "This message was deleted.",
                "ts": "1712345000.000100",
            },
            {"type": "message", "user": "U1", "text": "   ", "ts": "1712345100.000200"},
            {"type": "message", "user": "U2", "text": "Kept", "ts": "1712345200.000300"},
        ],
    )

    result = SlackJsonAdapter(path=str(export)).ingest()

    assert [unit.content for unit in result.units] == ["Kept"]


def test_slack_json_filters_entity_type_and_since(tmp_path):
    export = tmp_path / "general.json"
    _write_json(
        export,
        [
            {"type": "message", "user": "U1", "text": "Old", "ts": "1712300000.000001"},
            {"type": "message", "user": "U2", "text": "New", "ts": "1712500000.000002"},
        ],
    )

    skipped = SlackJsonAdapter(path=str(export)).ingest(entity_types=["markdown_note"])
    assert skipped.units == []
    assert skipped.edges == []

    result = SlackJsonAdapter(path=str(export)).ingest(
        since=SyncState(
            source_project="slack_json",
            source_entity_type="slack_message",
            last_sync_at=datetime.fromtimestamp(1712400000, tz=timezone.utc),
        )
    )
    assert [unit.content for unit in result.units] == ["New"]


def test_slack_json_adapter_is_registered():
    assert "slack_json" in list_adapters()
    adapter = get_adapter("slack_json", path="/tmp/slack/general")
    assert isinstance(adapter, SlackJsonAdapter)
    assert adapter.name == "slack_json"
