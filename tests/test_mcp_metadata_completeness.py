"""Tests for the MCP metadata completeness tool."""

from __future__ import annotations

import asyncio
import json

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def _insert_unit(
    store: Store,
    unit_id: str,
    *,
    source_project: SourceProject = SourceProject.MAX,
    source_entity_type: str = "insight",
    content_type: ContentType = ContentType.INSIGHT,
    metadata: dict | None = None,
) -> str:
    unit = KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=content_type,
        metadata=metadata or {},
    )
    return store.insert_unit(unit).id


def test_metadata_completeness_tool_is_listed_with_supported_filters():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "metadata_completeness")

    schema = tool.inputSchema
    assert schema["required"] == ["metadata_keys"]
    assert schema["properties"]["metadata_keys"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": "Dotted metadata paths to check, e.g. author or review.state",
    }
    assert schema["properties"]["source_project"]["type"] == "string"
    assert "max" in schema["properties"]["source_project"]["enum"]
    assert schema["properties"]["source_entity_type"]["type"] == "string"
    assert schema["properties"]["limit"]["minimum"] == 0


def test_metadata_completeness_tool_returns_store_summary_json(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _insert_unit(
        store,
        "max-insight-a",
        metadata={"doi": "10.1000/a", "review": {"state": "approved"}},
    )
    _insert_unit(store, "max-insight-b", metadata={})
    _insert_unit(
        store,
        "max-paper",
        source_entity_type="paper",
        metadata={},
    )
    _insert_unit(
        store,
        "forty-two-insight",
        source_project=SourceProject.FORTY_TWO,
        metadata={},
    )
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "metadata_completeness",
            {
                "metadata_keys": ["review.state", "doi"],
                "source_project": "max",
                "source_entity_type": "insight",
                "limit": 1,
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["total_units"] == 2
    assert payload["required_keys"] == ["doi", "review.state"]
    assert payload["source_project"] == "max"
    assert payload["source_entity_type"] == "insight"
    assert payload["present_counts"] == {"doi": 1, "review.state": 1}
    assert payload["missing_counts"] == {"doi": 1, "review.state": 1}
    assert payload["missing_unit_ids"] == {
        "doi": ["max-insight-b"],
        "review.state": ["max-insight-b"],
    }
    assert payload["keys"][0] == {
        "key": "doi",
        "present_count": 1,
        "missing_count": 1,
        "missing_unit_ids": ["max-insight-b"],
    }


def test_metadata_completeness_tool_validates_invalid_limit(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    Store(str(db_path)).close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "metadata_completeness",
            {"metadata_keys": ["doi"], "limit": True},
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "error": "limit must be a non-negative integer",
        "metadata_keys": ["doi"],
        "keys": [],
    }
