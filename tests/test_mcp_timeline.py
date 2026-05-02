"""Focused tests for the MCP timeline tool."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def _populate_timeline_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="jan-solar",
            source_entity_type="insight",
            title="January solar",
            content="Solar storage",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar"],
            created_at=datetime.fromisoformat("2026-01-15T10:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="feb-grid",
            source_entity_type="knowledge_node",
            title="February grid",
            content="Grid finding",
            content_type=ContentType.FINDING,
            tags=["energy", "grid"],
            created_at=datetime.fromisoformat("2026-02-05T10:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="feb-battery",
            source_entity_type="insight",
            title="February battery",
            content="Battery storage",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage"],
            created_at=datetime.fromisoformat("2026-02-20T10:00:00+00:00"),
        ),
    ]:
        store.insert_unit(unit)


@pytest.fixture
def timeline_store(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_timeline_graph(store)
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    return db_path


def test_timeline_tool_schema_and_successful_call(timeline_store):
    tools = asyncio.run(mcp_server.list_tools())
    timeline_tool = next(tool for tool in tools if tool.name == "timeline")

    assert timeline_tool.inputSchema["properties"]["bucket"]["enum"] == [
        "day",
        "week",
        "month",
        "year",
    ]
    assert timeline_tool.inputSchema["properties"]["field"]["enum"] == [
        "created_at",
        "ingested_at",
        "updated_at",
    ]
    assert set(timeline_tool.inputSchema["properties"]) >= {
        "bucket",
        "field",
        "start",
        "end",
        "source_project",
        "content_type",
        "tag",
    }

    response = asyncio.run(
        mcp_server.call_tool(
            "timeline",
            {
                "bucket": "month",
                "start": "2026-01-01T00:00:00+00:00",
                "end": "2026-02-28T23:59:59+00:00",
                "source_project": "max",
                "content_type": "insight",
                "tag": "energy",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["bucket"] == "month"
    assert payload["field"] == "created_at"
    assert payload["total"] == 2
    assert [item["bucket"] for item in payload["buckets"]] == ["2026-01", "2026-02"]
    assert payload["buckets"][1]["content_types"] == {"insight": 1}
    assert payload["buckets"][1]["source_projects"] == {"max": 1}
    assert payload["filters"] == {
        "source_project": "max",
        "content_type": "insight",
        "tag": "energy",
        "start": "2026-01-01T00:00:00+00:00",
        "end": "2026-02-28T23:59:59+00:00",
        "limit": None,
    }


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        (
            {"bucket": "quarter"},
            "invalid_bucket",
            "Unsupported timeline bucket",
        ),
        (
            {"field": "deleted_at"},
            "invalid_field",
            "Unsupported timeline field",
        ),
    ],
)
def test_timeline_tool_serializes_validation_errors(
    timeline_store,
    arguments,
    error,
    message,
):
    response = asyncio.run(mcp_server.call_tool("timeline", arguments))
    payload = json.loads(response[0].text)

    assert payload["error"] == error
    assert message in payload["message"]
    assert payload["arguments"] == {
        "bucket": arguments.get("bucket", "month"),
        "field": arguments.get("field", "created_at"),
        "start": None,
        "end": None,
        "limit": None,
        "source_project": None,
        "content_type": None,
        "tag": None,
    }
