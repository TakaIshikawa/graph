"""Tests for the ChatGPT JSON conversation adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.chatgpt_json import ChatGptJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def _conversation(conversation_id: str = "conv-1") -> dict:
    return {
        "id": conversation_id,
        "title": "Graph import ideas",
        "create_time": 1_700_000_000.0,
        "update_time": 1_700_000_300.0,
        "mapping": {
            "root": {
                "id": "root",
                "message": None,
                "parent": None,
                "children": ["user-1"],
            },
            "user-1": {
                "id": "user-1",
                "message": {
                    "id": "msg-user-1",
                    "author": {"role": "user"},
                    "create_time": 1_700_000_010.0,
                    "content": {"content_type": "text", "parts": ["Summarize graph imports."]},
                },
                "parent": "root",
                "children": ["assistant-1"],
            },
            "assistant-1": {
                "id": "assistant-1",
                "message": {
                    "id": "msg-assistant-1",
                    "author": {"role": "assistant"},
                    "create_time": 1_700_000_020.0,
                    "update_time": 1_700_000_025.0,
                    "content": {
                        "content_type": "text",
                        "parts": ["Use one unit per conversation."],
                    },
                },
                "parent": "user-1",
                "children": ["tool-1", "user-2"],
            },
            "tool-1": {
                "id": "tool-1",
                "message": {
                    "id": "msg-tool-1",
                    "author": {"role": "tool"},
                    "create_time": 1_700_000_030.0,
                    "content": {"content_type": "text", "parts": [""]},
                },
                "parent": "assistant-1",
                "children": [],
            },
            "user-2": {
                "id": "user-2",
                "message": {
                    "id": "msg-user-2",
                    "author": {"role": "user"},
                    "create_time": 1_700_000_040.0,
                    "content": {"content_type": "text", "parts": ["Include metadata too."]},
                },
                "parent": "assistant-1",
                "children": [],
            },
        },
    }


def test_ingests_conversations_json_with_readable_transcript_and_metadata(tmp_path):
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([_conversation()]), encoding="utf-8")

    result = ChatGptJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.CHATGPT_JSON
    assert unit.source_id == "chatgpt_json:conv-1"
    assert unit.source_entity_type == "chatgpt_conversation"
    assert unit.title == "Graph import ideas"
    assert unit.content == (
        "User: Summarize graph imports.\n\n"
        "Assistant: Use one unit per conversation.\n\n"
        "User: Include metadata too."
    )
    assert unit.metadata["conversation_id"] == "conv-1"
    assert unit.metadata["author_roles"] == ["user", "assistant"]
    assert unit.metadata["message_count"] == 3
    assert unit.metadata["message_ids"] == [
        "msg-user-1",
        "msg-assistant-1",
        "msg-user-2",
    ]
    assert unit.metadata["source_path"] == "conversations.json"
    assert unit.created_at == datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc)
    assert unit.updated_at == datetime.fromtimestamp(1_700_000_300.0, tz=timezone.utc)
    assert len(result.edges) == 2


def test_chatgpt_json_emits_message_reply_edges_without_message_units(tmp_path):
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([_conversation()]), encoding="utf-8")

    first = ChatGptJsonAdapter(path=str(export)).ingest()
    second = ChatGptJsonAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in first.units] == ["chatgpt_json:conv-1"]
    assert [(edge.relation, edge.source) for edge in first.edges] == [
        (EdgeRelation.REPLIES_TO, EdgeSource.SOURCE),
        (EdgeRelation.REPLIES_TO, EdgeSource.SOURCE),
    ]
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]
    assert [edge.created_at for edge in first.edges] == [edge.created_at for edge in second.edges]
    assert [(edge.metadata["child_node_id"], edge.metadata["parent_node_id"]) for edge in first.edges] == [
        ("assistant-1", "user-1"),
        ("user-2", "assistant-1"),
    ]
    assert first.edges[0].metadata["conversation_id"] == "conv-1"
    assert first.edges[0].metadata["child_role"] == "assistant"
    assert first.edges[0].metadata["parent_role"] == "user"
    assert all("tool-1" not in {edge.metadata["child_node_id"], edge.metadata["parent_node_id"]} for edge in first.edges)


def test_chatgpt_json_emits_attachment_units_with_parent_metadata(tmp_path):
    conversation = _conversation()
    conversation["mapping"]["user-1"]["message"]["metadata"] = {
        "attachments": [
            {
                "file_name": "notes.pdf",
                "mime_type": "application/pdf",
                "url": "https://files.example/notes.pdf",
            }
        ]
    }
    conversation["mapping"]["assistant-1"]["message"]["content"]["parts"].append(
        {"name": "chart.png", "mime_type": "image/png", "asset_pointer": "file-service://chart"}
    )
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([conversation]), encoding="utf-8")

    result = ChatGptJsonAdapter(path=str(export)).ingest()

    attachments = sorted(
        [unit for unit in result.units if unit.source_entity_type == "attachment"],
        key=lambda unit: unit.metadata["attachment_name"],
    )
    assert [unit.metadata["attachment_name"] for unit in attachments] == ["chart.png", "notes.pdf"]
    assert attachments[0].metadata["conversation_id"] == "conv-1"
    assert attachments[0].metadata["message_id"] == "msg-assistant-1"
    assert attachments[0].metadata["attachment_type"] == "image/png"
    assert attachments[0].tags == ["chatgpt", "attachment"]
    assert [unit.source_id for unit in ChatGptJsonAdapter(path=str(export)).ingest().units if unit.source_entity_type == "attachment"] == [
        unit.source_id for unit in attachments
    ]


def test_skips_malformed_and_empty_conversations_without_crashing(tmp_path):
    export = tmp_path / "mixed.json"
    malformed = tmp_path / "broken.json"
    export.write_text(
        json.dumps(
            [
                {"id": "empty", "title": "Empty", "mapping": {}},
                {"id": "blank", "title": "Blank", "mapping": {"n": {"message": {"content": {"parts": []}}}}},
                _conversation("usable"),
            ]
        ),
        encoding="utf-8",
    )
    malformed.write_text("{not json", encoding="utf-8")

    result = ChatGptJsonAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == ["chatgpt_json:usable"]


def test_ingests_directory_of_json_files_deterministically(tmp_path):
    first = tmp_path / "b.json"
    second = tmp_path / "a.json"
    first.write_text(json.dumps(_conversation("conv-b")), encoding="utf-8")
    second.write_text(json.dumps({"conversations": [_conversation("conv-a")]}), encoding="utf-8")

    result = ChatGptJsonAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "chatgpt_json:conv-a",
        "chatgpt_json:conv-b",
    ]


def test_registry_returns_chatgpt_json_adapter():
    adapter = get_adapter("chatgpt_json", path="/tmp/conversations.json")

    assert isinstance(adapter, ChatGptJsonAdapter)


def test_entity_type_and_since_filters(tmp_path):
    export = tmp_path / "conversations.json"
    export.write_text(json.dumps([_conversation()]), encoding="utf-8")
    adapter = ChatGptJsonAdapter(path=str(export))

    assert adapter.ingest(entity_types=["other"]).units == []
    filtered = adapter.ingest(
        since=SyncState(
            source_project="chatgpt_json",
            source_entity_type="chatgpt_conversation",
            last_sync_at=datetime.fromtimestamp(1_700_000_300.0, tz=timezone.utc),
        )
    )
    assert filtered.units == []
