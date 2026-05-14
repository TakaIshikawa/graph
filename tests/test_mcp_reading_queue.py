"""Focused tests for the MCP RAG reading queue tool."""

from __future__ import annotations

import asyncio
import json

import pytest

from graph.mcp import server as mcp_server
from graph.types.enums import EdgeRelation


NOW = "2026-05-02T00:00:00+00:00"


def _result(
    unit_id: str,
    title: str,
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> dict:
    return {
        "id": unit_id,
        "source_project": "max",
        "source_id": f"source-{unit_id}",
        "source_entity_type": "insight",
        "title": title,
        "content": f"{title} note",
        "content_type": "insight",
        "metadata": metadata or {},
        "tags": tags or [],
        "created_at": NOW,
        "updated_at": NOW,
    }


def test_graph_rag_reading_queue_tool_is_registered_with_schema():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "graph_rag_reading_queue")
    properties = tool.inputSchema["properties"]

    assert set(properties) >= {"results", "edges", "limit", "now"}
    assert properties["results"]["items"] == {"type": "object"}
    assert properties["edges"]["items"]["required"] == [
        "from_unit_id",
        "to_unit_id",
        "relation",
    ]
    assert properties["edges"]["items"]["properties"]["relation"]["enum"] == [
        relation.value for relation in EdgeRelation
    ]
    assert properties["limit"]["type"] == ["integer", "null"]
    assert properties["limit"]["minimum"] == 0


def test_graph_rag_reading_queue_returns_ordered_queue_with_reasons():
    response = asyncio.run(
        mcp_server.call_tool(
            "graph_rag_reading_queue",
            {
                "results": [
                    _result(
                        "read-low",
                        "Read Low",
                        metadata={
                            "priority": "low",
                            "read_status": "read",
                            "last_read_at": NOW,
                        },
                    ),
                    _result(
                        "high-unread",
                        "High Unread",
                        metadata={"priority": "high", "read_status": "unread"},
                    ),
                    _result(
                        "medium-unread",
                        "Medium Unread",
                        metadata={"priority": "medium", "read_status": "unread"},
                        tags=["review"],
                    ),
                ],
                "edges": [
                    {
                        "id": "edge-high-medium",
                        "from_unit_id": "high-unread",
                        "to_unit_id": "medium-unread",
                        "relation": "references",
                    }
                ],
                "limit": 2,
                "now": NOW,
            },
        )
    )
    payload = json.loads(response[0].text)

    assert [item["unit_id"] for item in payload["queue"]] == [
        "medium-unread",
        "high-unread",
    ]
    assert [item["order"] for item in payload["queue"]] == [1, 2]
    assert payload["queue"][0]["reasons"] == [
        "unread",
        "priority",
        "never read",
        "review tag",
    ]
    assert payload["queue"][0]["inbound_reference_count"] == 1
    assert payload["queue"][1]["reason"] == "unread; high priority; never read"
    assert payload["stats"]["total_units"] == 3
    assert payload["stats"]["queued_units"] == 2
    assert payload["stats"]["omitted_units"] == 1
    assert payload["stats"]["edge_boosted_units"] == 1
    assert payload["options"]["limit"] == 2


def test_graph_rag_reading_queue_empty_input_returns_empty_queue():
    response = asyncio.run(mcp_server.call_tool("graph_rag_reading_queue", {}))
    payload = json.loads(response[0].text)

    assert payload["queue"] == []
    assert payload["units"] == []
    assert payload["stats"]["total_units"] == 0
    assert payload["stats"]["queued_units"] == 0
    assert payload["stats"]["omitted_units"] == 0
    assert payload["stats"]["limit"] == 20
    assert payload["options"]["limit"] == 20


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"limit": -1}, "limit must be a non-negative integer or null"),
        ({"limit": True}, "limit must be a non-negative integer or null"),
        ({"now": 123}, "now must be an ISO-8601 timestamp string"),
        ({"now": "not-a-date"}, "now must be an ISO-8601 timestamp string"),
        ({"results": ["bad"]}, "results[0] must be an object"),
        ({"edges": ["bad"]}, "edges[0] must be an object"),
    ],
)
def test_graph_rag_reading_queue_serializes_invalid_arguments(arguments, message):
    response = asyncio.run(mcp_server.call_tool("graph_rag_reading_queue", arguments))
    payload = json.loads(response[0].text)

    assert payload["error"] == "invalid_reading_queue_request"
    assert payload["message"] == message
    assert payload["arguments"]["limit"] == arguments.get("limit", 20)
