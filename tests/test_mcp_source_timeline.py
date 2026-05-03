"""Focused tests for the MCP source timeline tool."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def _unit(
    unit_id: str,
    title: str,
    *,
    source_project: SourceProject,
    created_at: str,
    tags: list[str],
    content_type: ContentType = ContentType.INSIGHT,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} solar grid storage note",
        content_type=content_type,
        tags=tags,
        created_at=datetime.fromisoformat(created_at),
    )


def _populate_source_timeline_graph(store: Store) -> None:
    for unit in [
        _unit(
            "unit-jan-max",
            "January solar",
            source_project=SourceProject.MAX,
            created_at="2026-01-15T10:00:00+00:00",
            tags=["energy", "solar"],
        ),
        _unit(
            "unit-feb-max",
            "February storage",
            source_project=SourceProject.MAX,
            created_at="2026-02-20T10:00:00+00:00",
            tags=["energy", "storage"],
        ),
        _unit(
            "unit-feb-presence",
            "February artifact",
            source_project=SourceProject.PRESENCE,
            created_at="2026-02-05T10:00:00+00:00",
            tags=["energy", "grid"],
            content_type=ContentType.ARTIFACT,
        ),
        _unit(
            "unit-mar-forty-two",
            "March unrelated",
            source_project=SourceProject.FORTY_TWO,
            created_at="2026-03-10T10:00:00+00:00",
            tags=["archive"],
            content_type=ContentType.FINDING,
        ),
    ]:
        store.insert_unit(unit)


@pytest.fixture
def source_timeline_store(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_source_timeline_graph(store)
    store.rebuild_fts_index()
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))


def test_source_timeline_tool_is_registered_with_schema():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "source_timeline")
    properties = tool.inputSchema["properties"]

    assert set(properties) >= {
        "query",
        "mode",
        "sort",
        "result_limit",
        "bucket",
        "limit",
        "source_project",
        "content_type",
        "tag",
        "created_after",
        "created_before",
    }
    assert properties["bucket"]["enum"] == ["day", "week", "month", "year"]
    assert properties["bucket"]["default"] == "month"
    assert properties["result_limit"]["default"] == 100
    assert properties["result_limit"]["minimum"] == 0
    assert properties["mode"]["enum"] == ["hybrid", "semantic", "fulltext"]


def test_source_timeline_returns_filtered_store_units(source_timeline_store):
    response = asyncio.run(
        mcp_server.call_tool(
            "source_timeline",
            {
                "bucket": "month",
                "tag": "energy",
                "created_before": "2026-02-28T23:59:59+00:00",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["buckets"] == [
        {
            "bucket": "2026-01",
            "start": "2026-01-01",
            "sources": {"max": 1},
            "total": 1,
        },
        {
            "bucket": "2026-02",
            "start": "2026-02-01",
            "sources": {"max": 1, "presence": 1},
            "total": 2,
        },
    ]
    assert payload["sources"] == ["max", "presence"]
    assert payload["stats"]["result_count"] == 3
    assert payload["stats"]["bucket"] == "month"
    assert payload["filters"] == {
        "tag": "energy",
        "created_before": "2026-02-28T23:59:59+00:00",
    }
    assert payload["selection"]["source"] == "store"
    assert payload["selection"]["result_count"] == 3


def test_source_timeline_uses_query_search_and_bucket_limit(source_timeline_store):
    response = asyncio.run(
        mcp_server.call_tool(
            "source_timeline",
            {
                "query": "solar",
                "mode": "fulltext",
                "bucket": "month",
                "limit": 1,
                "source_project": "max",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert [bucket["bucket"] for bucket in payload["buckets"]] == ["2026-01"]
    assert payload["stats"]["candidate_count"] == 2
    assert payload["stats"]["included_count"] == 1
    assert payload["stats"]["omitted_buckets"] == 1
    assert payload["selection"]["source"] == "search"
    assert payload["selection"]["query"] == "solar"
    assert set(payload["selection"]["result_ids"]) == {"unit-jan-max", "unit-feb-max"}
    assert payload["filters"] == {"source_project": "max"}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"bucket": "quarter"}, "bucket must be one of: day, month, week, year"),
        ({"result_limit": True}, "result_limit must be a non-negative integer"),
        ({"limit": -1}, "limit must be a non-negative integer"),
        (
            {
                "created_after": "2026-03-01T00:00:00+00:00",
                "created_before": "2026-01-01T00:00:00+00:00",
            },
            "created_after must be on or before created_before.",
        ),
    ],
)
def test_source_timeline_serializes_invalid_arguments(
    source_timeline_store,
    arguments,
    message,
):
    response = asyncio.run(mcp_server.call_tool("source_timeline", arguments))
    payload = json.loads(response[0].text)

    assert payload["error"] == "invalid_source_timeline_request"
    assert payload["message"] == message
    assert payload["arguments"]["bucket"] == arguments.get("bucket", "month")
    assert payload["arguments"]["result_limit"] == arguments.get("result_limit", 100)
