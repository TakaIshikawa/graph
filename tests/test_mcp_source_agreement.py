"""Focused tests for the MCP RAG source agreement tool."""

from __future__ import annotations

import asyncio
import json

import pytest

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def _unit(
    source_id: str,
    source_project: SourceProject,
    title: str,
    content: str,
    tags: list[str],
) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=source_project,
        source_id=source_id,
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def _populate_source_agreement_graph(store: Store) -> dict[str, str]:
    ids = {}
    for key, unit in [
        (
            "solar_max",
            _unit(
                "solar-max",
                SourceProject.MAX,
                "Solar storage plan",
                "Grid storage improves solar reliability.",
                ["Solar", "Storage"],
            ),
        ),
        (
            "solar_presence",
            _unit(
                "solar-presence",
                SourceProject.PRESENCE,
                "Solar finance update",
                "Storage finance improves adoption.",
                ["solar", "finance"],
            ),
        ),
        (
            "grid",
            _unit(
                "grid",
                SourceProject.FORTY_TWO,
                "Grid storage note",
                "Storage supports grid planning.",
                ["Grid", "Storage"],
            ),
        ),
    ]:
        ids[key] = store.insert_unit(unit).id
    return ids


@pytest.fixture
def source_agreement_store(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    ids = _populate_source_agreement_graph(store)
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    return ids


def test_source_agreement_tool_is_registered_with_schema(source_agreement_store):
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "source_agreement")

    assert set(tool.inputSchema["properties"]) >= {
        "claim",
        "query",
        "results",
        "unit_ids",
        "min_source_count",
        "limit",
        "min_term_length",
    }
    assert tool.inputSchema["properties"]["unit_ids"]["items"] == {"type": "string"}
    assert tool.inputSchema["properties"]["min_source_count"]["minimum"] == 1


def test_source_agreement_tool_scores_fetched_unit_ids(source_agreement_store):
    response = asyncio.run(
        mcp_server.call_tool(
            "source_agreement",
            {
                "claim": "Solar storage improves grid reliability.",
                "unit_ids": [
                    source_agreement_store["solar_max"],
                    source_agreement_store["solar_presence"],
                    source_agreement_store["grid"],
                ],
                "min_source_count": 2,
                "limit": 5,
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["status"] == "ok"
    assert payload["claim"] == "Solar storage improves grid reliability."
    assert payload["stats"] == {
        "result_count": 3,
        "source_count": 3,
        "min_source_count": 2,
        "missing_unit_ids": [],
        "limit": 5,
    }
    storage = next(
        row
        for row in payload["scores"]
        if row["evidence_type"] == "term" and row["evidence_key"] == "storage"
    )
    assert storage["source_count"] == 3
    assert storage["agreement_score"] == 1.0
    assert storage["supporting_source_projects"] == [
        "forty_two",
        "max",
        "presence",
    ]


def test_source_agreement_tool_empty_input_returns_low_information(
    source_agreement_store,
):
    response = asyncio.run(mcp_server.call_tool("source_agreement", {}))
    payload = json.loads(response[0].text)

    assert payload == {
        "claim": None,
        "status": "low_information",
        "reason": "No result records or resolvable unit_ids were provided.",
        "scores": [],
        "stats": {
            "result_count": 0,
            "source_count": 0,
            "min_source_count": 2,
            "missing_unit_ids": [],
        },
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"min_source_count": 0}, "min_source_count must be a positive integer"),
        ({"limit": -1}, "limit must be a non-negative integer or None"),
        ({"results": ["bad"]}, "results[0] must be an object"),
        ({"unit_ids": [123]}, "unit_ids must be an array of strings"),
    ],
)
def test_source_agreement_tool_serializes_invalid_arguments(
    source_agreement_store,
    arguments,
    message,
):
    response = asyncio.run(mcp_server.call_tool("source_agreement", arguments))
    payload = json.loads(response[0].text)

    assert payload["error"] == "invalid_source_agreement_request"
    assert payload["message"] == message
    assert payload["arguments"]["min_source_count"] == arguments.get(
        "min_source_count",
        2,
    )
