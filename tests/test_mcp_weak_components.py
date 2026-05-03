"""Focused tests for the MCP weak component summary tool."""

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
    source_project: SourceProject = SourceProject.MAX,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} note",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def _edge(edge_id: str, from_unit_id: str, to_unit_id: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.RELATES_TO,
    )


def _populate_weak_component_graph(store: Store) -> None:
    for unit in [
        _unit("unit-alpha", "Alpha", tags=["energy", "solar"]),
        _unit(
            "unit-beta",
            "Beta",
            source_project=SourceProject.FORTY_TWO,
            tags=["energy", "grid"],
        ),
        _unit("unit-gamma", "Gamma", tags=["storage"]),
        _unit("unit-delta", "Delta", source_project=SourceProject.PRESENCE),
        _unit("unit-isolated", "Isolated", tags=["solo"]),
    ]:
        store.insert_unit(unit)
    for edge in [
        _edge("edge-beta-alpha", "unit-beta", "unit-alpha"),
        _edge("edge-gamma-delta", "unit-gamma", "unit-delta"),
    ]:
        store.insert_edge(edge)


@pytest.fixture
def weak_components_store(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_weak_component_graph(store)
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))


def test_weak_component_summary_tool_is_registered_with_schema():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "weak_component_summary")
    properties = tool.inputSchema["properties"]

    assert properties["limit"]["default"] == 20
    assert properties["limit"]["minimum"] == 0
    assert properties["representative_limit"]["default"] == 3
    assert properties["representative_limit"]["minimum"] == 0


def test_weak_component_summary_returns_component_summaries(weak_components_store):
    response = asyncio.run(
        mcp_server.call_tool(
            "weak_component_summary",
            {"limit": None, "representative_limit": 2},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["node_count"] == 5
    assert payload["edge_count"] == 2
    assert payload["component_count"] == 3
    assert payload["isolated_component_count"] == 1
    assert payload["components"] == [
        {
            "component_id": "component-001",
            "unit_ids": ["unit-alpha", "unit-beta"],
            "size": 2,
            "representative_unit_ids": ["unit-alpha", "unit-beta"],
            "representative_titles": ["Alpha", "Beta"],
            "source_project_counts": {"max": 1, "forty_two": 1},
            "source_breakdown": {"max": 1, "forty_two": 1},
            "tag_breakdown": {"energy": 2, "grid": 1, "solar": 1},
            "tag_counts": {"energy": 2, "grid": 1, "solar": 1},
            "internal_edge_count": 1,
            "isolated": False,
        },
        {
            "component_id": "component-002",
            "unit_ids": ["unit-delta", "unit-gamma"],
            "size": 2,
            "representative_unit_ids": ["unit-delta", "unit-gamma"],
            "representative_titles": ["Delta", "Gamma"],
            "source_project_counts": {"presence": 1, "max": 1},
            "source_breakdown": {"presence": 1, "max": 1},
            "tag_breakdown": {"storage": 1},
            "tag_counts": {"storage": 1},
            "internal_edge_count": 1,
            "isolated": False,
        },
        {
            "component_id": "component-003",
            "unit_ids": ["unit-isolated"],
            "size": 1,
            "representative_unit_ids": ["unit-isolated"],
            "representative_titles": ["Isolated"],
            "source_project_counts": {"max": 1},
            "source_breakdown": {"max": 1},
            "tag_breakdown": {"solo": 1},
            "tag_counts": {"solo": 1},
            "internal_edge_count": 0,
            "isolated": True,
        },
    ]


def test_weak_component_summary_applies_limits(weak_components_store):
    response = asyncio.run(
        mcp_server.call_tool(
            "weak_component_summary",
            {"limit": 1, "representative_limit": 1},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["component_count"] == 3
    assert len(payload["components"]) == 1
    assert payload["components"][0]["component_id"] == "component-001"
    assert payload["components"][0]["size"] == 2
    assert payload["components"][0]["representative_unit_ids"] == ["unit-alpha"]
    assert payload["components"][0]["representative_titles"] == ["Alpha"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"limit": -1}, "limit must be a non-negative integer or None."),
        ({"limit": True}, "limit must be a non-negative integer or None."),
        (
            {"representative_limit": -1},
            "representative_limit must be a non-negative integer.",
        ),
    ],
)
def test_weak_component_summary_serializes_invalid_arguments(
    weak_components_store,
    arguments,
    message,
):
    response = asyncio.run(mcp_server.call_tool("weak_component_summary", arguments))
    payload = json.loads(response[0].text)

    assert payload["error"] == "invalid_weak_component_summary_request"
    assert payload["message"] == message
    assert payload["arguments"]["limit"] == arguments.get("limit", 20)
    assert payload["arguments"]["representative_limit"] == arguments.get(
        "representative_limit", 3
    )
