"""Focused tests for the MCP RAG context gap tool."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def _populate_context_gap_graph(store: Store) -> dict[str, str]:
    ids = {}
    for key, unit in [
        (
            "jan_solar",
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="jan-solar",
                source_entity_type="insight",
                title="January solar storage",
                content="Solar storage context for planning.",
                content_type=ContentType.INSIGHT,
                tags=["energy", "solar"],
                metadata={"domain": "energy.example"},
                created_at=datetime.fromisoformat("2026-01-15T10:00:00+00:00"),
                updated_at=datetime.fromisoformat("2026-01-15T10:00:00+00:00"),
            ),
        ),
        (
            "feb_battery",
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="feb-battery",
                source_entity_type="insight",
                title="February battery storage",
                content="Battery storage context for planning.",
                content_type=ContentType.INSIGHT,
                tags=["energy", "storage"],
                metadata={"domain": "battery.example"},
                created_at=datetime.fromisoformat("2026-02-20T10:00:00+00:00"),
                updated_at=datetime.fromisoformat("2026-02-20T10:00:00+00:00"),
            ),
        ),
        (
            "grid",
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="grid",
                source_entity_type="knowledge_node",
                title="Grid finding",
                content="Grid reliability context.",
                content_type=ContentType.FINDING,
                tags=["energy", "grid"],
                created_at=datetime.fromisoformat("2026-02-05T10:00:00+00:00"),
                updated_at=datetime.fromisoformat("2026-02-05T10:00:00+00:00"),
            ),
        ),
    ]:
        stored = store.insert_unit(unit)
        store.fts_index_unit(stored)
        ids[key] = stored.id
    return ids


@pytest.fixture
def context_gap_store(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    ids = _populate_context_gap_graph(store)
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    return ids


def test_context_gaps_tool_schema(context_gap_store):
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "context_gaps")

    assert tool.inputSchema["required"] == ["query"]
    assert tool.inputSchema["properties"]["mode"]["enum"] == [
        "hybrid",
        "semantic",
        "fulltext",
    ]
    assert set(tool.inputSchema["properties"]) >= {
        "query",
        "limit",
        "required_facets",
        "min_sources",
        "min_recent_items",
        "recency_window_days",
        "now",
    }


def test_context_gaps_tool_returns_gap_report(context_gap_store):
    response = asyncio.run(
        mcp_server.call_tool(
            "context_gaps",
            {
                "query": "storage",
                "limit": 10,
                "required_facets": {
                    "source_projects": ["max", "forty_two"],
                    "tags": ["energy", "storage"],
                },
                "min_sources": 2,
                "min_recent_items": 2,
                "recency_window_days": 15,
                "now": "2026-03-01T00:00:00+00:00",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["query"] == "storage"
    assert payload["mode"] == "fulltext"
    assert payload["search"]["result_count"] == 2
    assert set(payload["search"]["result_ids"]) == {
        context_gap_store["jan_solar"],
        context_gap_store["feb_battery"],
    }
    assert payload["coverage"]["result_count"] == 2
    assert payload["coverage"]["source_projects"] == [
        {"value": "max", "count": 2}
    ]
    assert payload["coverage"]["recency"]["recent_count"] == 1
    assert payload["coverage"]["recency"]["window_days"] == 15
    assert [gap["type"] for gap in payload["gaps"]] == [
        "source_diversity",
        "missing_required_source_projects",
        "recency",
    ]
    assert set(payload["representative_result_ids"]["source_projects"]["max"]) == {
        context_gap_store["jan_solar"],
        context_gap_store["feb_battery"],
    }


def test_context_gaps_tool_empty_search_results_are_structured(context_gap_store):
    response = asyncio.run(
        mcp_server.call_tool(
            "context_gaps",
            {
                "query": "zzzzmissing",
                "required_facets": {"tags": ["storage"]},
                "min_sources": 0,
                "min_recent_items": 1,
                "now": "2026-03-01T00:00:00+00:00",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["search"] == {"result_count": 0, "result_ids": []}
    assert payload["coverage"]["result_count"] == 0
    assert [gap["type"] for gap in payload["gaps"]] == [
        "empty_results",
        "missing_required_tags",
        "recency",
    ]
    assert payload["suggestions"][0] == (
        "Run a broader retrieval before generating an answer."
    )
