"""Focused tests for the MCP orphan unit analysis tool."""

from __future__ import annotations

import asyncio
import json

import pytest

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _unit(
    unit_id: str,
    title: str,
    *,
    tags: list[str] | None = None,
    source_project: SourceProject = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=content_type,
        tags=tags or [],
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def _populate_orphan_graph(store: Store) -> None:
    for unit in [
        _unit("unit-alpha", "Alpha", tags=["energy", "solar"]),
        _unit("unit-beta", "Beta", tags=["energy", "grid"]),
        _unit("unit-gamma", "Gamma", tags=["grid", "storage"]),
        _unit(
            "unit-isolated",
            "Isolated",
            tags=["archive"],
            source_project=SourceProject.PRESENCE,
            content_type=ContentType.ARTIFACT,
        ),
    ]:
        store.insert_unit(unit)

    for edge in [
        _edge("edge-alpha-beta", "unit-alpha", "unit-beta"),
        _edge("edge-beta-gamma", "unit-beta", "unit-gamma"),
    ]:
        store.insert_edge(edge)


@pytest.fixture
def orphan_store(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_orphan_graph(store)
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))


def test_analyze_orphan_units_tool_is_registered_with_schema():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "analyze_orphan_units")
    properties = tool.inputSchema["properties"]

    assert set(properties) >= {"min_degree", "include_metadata"}
    assert properties["min_degree"]["default"] == 1
    assert properties["min_degree"]["minimum"] == 0
    assert properties["include_metadata"]["type"] == "boolean"
    assert properties["include_metadata"]["default"] is True


def test_analyze_orphan_units_empty_store_returns_empty_payload(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    Store(str(db_path)).close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("analyze_orphan_units", {}))
    payload = json.loads(response[0].text)

    assert payload == {
        "min_degree": 1,
        "node_count": 0,
        "edge_count": 0,
        "candidate_count": 0,
        "include_metadata": True,
        "units": [],
    }


def test_analyze_orphan_units_returns_structured_degree_data(orphan_store):
    response = asyncio.run(
        mcp_server.call_tool("analyze_orphan_units", {"min_degree": 2})
    )
    payload = json.loads(response[0].text)

    assert payload["min_degree"] == 2
    assert payload["node_count"] == 4
    assert payload["edge_count"] == 2
    assert payload["candidate_count"] == 3
    assert [
        (
            unit["unit_id"],
            unit["degree"],
            unit["in_degree"],
            unit["out_degree"],
        )
        for unit in payload["units"]
    ] == [
        ("unit-isolated", 0, 0, 0),
        ("unit-alpha", 1, 0, 1),
        ("unit-gamma", 1, 1, 0),
    ]
    assert payload["units"][0]["source_project"] == "presence"
    assert payload["units"][0]["content_type"] == "artifact"
    assert payload["units"][1]["suggested_neighboring_tags"] == [
        {"tag": "grid", "neighbor_count": 1}
    ]


def test_analyze_orphan_units_honors_include_metadata_false(orphan_store):
    response = asyncio.run(
        mcp_server.call_tool(
            "analyze_orphan_units",
            {"min_degree": 2, "include_metadata": False},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["include_metadata"] is False
    assert payload["units"][0] == {
        "unit_id": "unit-isolated",
        "title": "Isolated",
        "degree": 0,
        "in_degree": 0,
        "out_degree": 0,
    }
    assert all("tags" not in unit for unit in payload["units"])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"min_degree": -1}, "min_degree must be a non-negative integer."),
        ({"min_degree": True}, "min_degree must be a non-negative integer."),
        ({"min_degree": "many"}, "min_degree must be a non-negative integer."),
    ],
)
def test_analyze_orphan_units_serializes_invalid_arguments(
    orphan_store,
    arguments,
    message,
):
    response = asyncio.run(mcp_server.call_tool("analyze_orphan_units", arguments))
    payload = json.loads(response[0].text)

    assert payload == {
        "error": "invalid_orphan_units_request",
        "message": message,
        "arguments": {
            "min_degree": arguments["min_degree"],
            "include_metadata": True,
        },
    }
