"""Focused tests for the MCP tag normalization tool."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from graph.mcp import server as mcp_server
from graph.rag import suggest_tag_normalizations
from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


NOW = datetime(2026, 5, 2, 0, 0, tzinfo=timezone.utc)


def _unit(unit_id: str, title: str, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
        tags=tags,
        created_at=NOW,
        updated_at=NOW,
    )


def _record(unit_id: str, title: str, tags: list[str]) -> dict:
    return {
        "id": unit_id,
        "source_project": "max",
        "source_id": f"source-{unit_id}",
        "source_entity_type": "insight",
        "title": title,
        "content": f"{title} note",
        "content_type": "insight",
        "tags": tags,
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
    }


def test_suggest_tag_normalizations_tool_is_registered_with_schema():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "suggest_tag_normalizations")
    properties = tool.inputSchema["properties"]

    assert set(properties) >= {
        "units",
        "results",
        "min_count",
        "min_similarity",
        "limit",
    }
    assert properties["units"]["items"] == {"type": "object"}
    assert properties["results"]["items"] == {"type": "object"}
    assert properties["min_count"]["minimum"] == 1
    assert properties["min_similarity"]["minimum"] == 0
    assert properties["min_similarity"]["maximum"] == 1
    assert properties["limit"]["type"] == ["integer", "null"]


def test_suggest_tag_normalizations_matches_rag_helper_for_explicit_results():
    units = [
        _unit("unit-a", "Agent hyphen", ["ai-agent"]),
        _unit("unit-b", "Agent underscore", ["ai_agent"]),
        _unit("unit-c", "Agent plural", ["AI Agents"]),
        _unit("unit-d", "Storage", ["storage"]),
    ]
    expected = suggest_tag_normalizations(units, min_similarity=0.75, limit=None)

    response = asyncio.run(
        mcp_server.call_tool(
            "suggest_tag_normalizations",
            {
                "results": [
                    _record(unit.id, unit.title, unit.tags)
                    for unit in units
                ],
                "min_similarity": 0.75,
                "limit": None,
            },
        )
    )
    payload = json.loads(response[0].text)

    core = [
        {
            "canonical_tag": suggestion["canonical_tag"],
            "variants": suggestion["variants"],
            "counts": suggestion["counts"],
            "similarity": suggestion["similarity"],
            "affected_unit_ids": suggestion["affected_unit_ids"],
        }
        for suggestion in payload["suggestions"]
    ]
    assert core == expected
    assert payload["suggestions"][0]["confidence"] == expected[0]["similarity"]
    assert payload["suggestions"][0]["score"] == pytest.approx(
        expected[0]["similarity"] * sum(expected[0]["counts"].values())
    )
    assert payload["stats"] == {
        "unit_count": 4,
        "suggestion_count": len(expected),
        "source": "explicit",
    }
    assert payload["options"] == {
        "min_count": 1,
        "min_similarity": 0.75,
        "limit": None,
    }


def test_suggest_tag_normalizations_reads_units_from_configured_store(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    store.insert_unit(_unit("store-a", "Agent hyphen", ["ai-agent"]))
    store.insert_unit(_unit("store-b", "Agent underscore", ["ai_agent"]))
    store.insert_unit(_unit("store-c", "Storage", ["storage"]))
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "suggest_tag_normalizations",
            {"min_similarity": 0.75},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["stats"]["source"] == "store"
    assert payload["stats"]["unit_count"] == 3
    assert payload["suggestions"] == [
        {
            "canonical_tag": "ai-agent",
            "variants": ["ai_agent"],
            "counts": {"ai-agent": 1, "ai_agent": 1},
            "similarity": 1.0,
            "affected_unit_ids": ["store-a", "store-b"],
            "confidence": 1.0,
            "score": 2.0,
        }
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"min_count": 0}, "min_count must be a positive integer"),
        ({"min_count": True}, "min_count must be a positive integer"),
        ({"min_similarity": 1.2}, "min_similarity must be a number between 0 and 1"),
        ({"min_similarity": False}, "min_similarity must be a number between 0 and 1"),
        ({"limit": 0}, "limit must be a positive integer"),
        ({"results": ["bad"]}, "results[0] must be an object"),
        ({"results": [{"tags": "ai-agent"}]}, "results[0].tags must be an array of strings"),
    ],
)
def test_suggest_tag_normalizations_serializes_invalid_arguments(arguments, message):
    response = asyncio.run(mcp_server.call_tool("suggest_tag_normalizations", arguments))
    payload = json.loads(response[0].text)

    assert payload["error"] == "invalid_tag_normalization_request"
    assert payload["message"] == message
    assert payload["arguments"]["min_count"] == arguments.get("min_count", 1)
    assert payload["arguments"]["min_similarity"] == arguments.get("min_similarity", 0.82)
    assert payload["arguments"]["limit"] == arguments.get("limit", 50)
