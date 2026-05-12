from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.slack_json import SlackJsonAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
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
        [{"type": "message", "user": "U1", "text": "First", "ts": "1712300000.000001"}],
    )
    _write_json(
        channel / "2024-04-06.json",
        [{"type": "message", "user": "U2", "text": "Second", "ts": "1712400000.000002"}],
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

    root, reply = [unit for unit in result.units if unit.source_entity_type == "slack_message"]
    assert root.metadata["thread_ts"] == "1712345000.000100"
    assert root.metadata["is_thread_reply"] is False
    assert root.metadata["reply_count"] == 1
    assert reply.metadata["thread_ts"] == "1712345000.000100"
    assert reply.metadata["is_thread_reply"] is True


def test_slack_json_emits_thread_reply_edges(tmp_path):
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
                "reply_count": 2,
            },
            {
                "type": "message",
                "user": "U2",
                "text": "First reply",
                "ts": "1712345100.000200",
                "thread_ts": "1712345000.000100",
            },
            {
                "type": "message",
                "user": "U3",
                "text": "Second reply",
                "ts": "1712345200.000300",
                "thread_ts": "1712345000.000100",
            },
            {
                "type": "message",
                "user": "U4",
                "text": "Orphan reply",
                "ts": "1712345300.000400",
                "thread_ts": "1712340000.000001",
            },
        ],
    )

    first = SlackJsonAdapter(path=str(channel)).ingest()
    second = SlackJsonAdapter(path=str(channel)).ingest()

    root_id = "slack_json:design:1712345000.000100"
    reply_edges = [edge for edge in first.edges if edge.relation == EdgeRelation.REPLIES_TO]
    assert [
        (edge.from_unit_id, edge.to_unit_id, edge.relation, edge.source) for edge in reply_edges
    ] == [
        (
            "slack_json:design:1712345100.000200",
            root_id,
            EdgeRelation.REPLIES_TO,
            EdgeSource.SOURCE,
        ),
        (
            "slack_json:design:1712345200.000300",
            root_id,
            EdgeRelation.REPLIES_TO,
            EdgeSource.SOURCE,
        ),
    ]
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]
    assert [edge.created_at for edge in first.edges] == [edge.created_at for edge in second.edges]
    assert all(edge.from_unit_id != edge.to_unit_id for edge in first.edges)
    assert reply_edges[0].metadata["thread_ts"] == "1712345000.000100"


def test_slack_json_emits_thread_summary_units(tmp_path):
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
                "reply_count": 2,
            },
            {
                "type": "message",
                "user": "U2",
                "text": "First reply",
                "ts": "1712345100.000200",
                "thread_ts": "1712345000.000100",
            },
            {
                "type": "message",
                "user": "U1",
                "text": "Second reply",
                "ts": "1712345200.000300",
                "thread_ts": "1712345000.000100",
            },
        ],
    )

    result = SlackJsonAdapter(path=str(channel)).ingest()

    summary = next(unit for unit in result.units if unit.source_entity_type == "slack_thread")
    assert summary.source_id == "slack_json:design:1712345000.000100:thread_summary"
    assert summary.metadata["channel"] == "design"
    assert summary.metadata["thread_ts"] == "1712345000.000100"
    assert summary.metadata["participant_count"] == 2
    assert summary.metadata["reply_count"] == 2
    assert (
        summary.metadata["started_at"]
        == datetime.fromtimestamp(1712345000.0001, tz=timezone.utc).isoformat()
    )
    assert (
        summary.metadata["ended_at"]
        == datetime.fromtimestamp(1712345200.0003, tz=timezone.utc).isoformat()
    )
    assert summary.metadata["message_unit_ids"] == [
        "slack_json:design:1712345000.000100",
        "slack_json:design:1712345100.000200",
        "slack_json:design:1712345200.000300",
    ]
    contains_edges = [edge for edge in result.edges if edge.relation == EdgeRelation.CONTAINS]
    assert len(contains_edges) == 3
    assert {edge.from_unit_id for edge in contains_edges} == {summary.source_id}


def test_slack_json_emits_reaction_units_without_losing_message(tmp_path):
    export = tmp_path / "general.json"
    _write_json(
        export,
        [
            {
                "type": "message",
                "user": "U1",
                "text": "React to this",
                "ts": "1712345000.000100",
                "reactions": [
                    {"name": "eyes", "count": 2, "users": ["U2", "U3"]},
                    {"name": "thumbsup", "count": 1, "users": ["U4"]},
                ],
            }
        ],
    )

    result = SlackJsonAdapter(path=str(export)).ingest()

    message = next(unit for unit in result.units if unit.source_entity_type == "slack_message")
    reactions = [unit for unit in result.units if unit.source_entity_type == "reaction"]
    assert message.metadata["reaction_count"] == 3
    assert [unit.metadata["reaction_name"] for unit in reactions] == ["eyes", "thumbsup"]
    assert reactions[0].metadata["message_ts"] == "1712345000.000100"
    assert reactions[0].metadata["reaction_count"] == 2
    assert reactions[0].metadata["reacting_users"] == ["U2", "U3"]
    assert reactions[0].metadata["channel"] == "general"
    assert reactions[0].metadata["source_file"] == "general.json"
    assert reactions[0].tags[:2] == ["slack", "reaction"]
    reference_edges = [edge for edge in result.edges if edge.relation == EdgeRelation.REFERENCES]
    assert len(reference_edges) == 2
    assert {edge.from_unit_id for edge in reference_edges} == {unit.source_id for unit in reactions}
    assert {edge.to_unit_id for edge in reference_edges} == {message.source_id}


def test_slack_json_reaction_entity_filtering(tmp_path):
    export = tmp_path / "general.json"
    _write_json(
        export,
        [
            {
                "type": "message",
                "user": "U1",
                "text": "React to this",
                "ts": "1712345000.000100",
                "reactions": [{"name": "eyes", "count": 1, "users": ["U2"]}],
            }
        ],
    )

    reactions_only = SlackJsonAdapter(path=str(export)).ingest(entity_types=["reaction"])
    assert [unit.source_entity_type for unit in reactions_only.units] == ["reaction"]
    assert reactions_only.edges == []


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
