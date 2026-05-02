from __future__ import annotations

import asyncio
import json

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def _unit(source_id: str, title: str, content: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=SourceProject.MAX,
        source_id=source_id,
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=["mcp", "collection"],
    )


def test_graph_collection_members_tool_is_registered_with_schema():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "graph_collection_members")

    assert tool.inputSchema["required"] == ["collection_name"]
    assert tool.inputSchema["properties"]["collection_name"]["type"] == "string"
    assert tool.inputSchema["properties"]["limit"] == {
        "type": "integer",
        "minimum": 1,
        "description": "Maximum members to return",
    }


def test_graph_collection_members_returns_collection_and_member_summaries(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    alpha = store.insert_unit(_unit("alpha", "Alpha", "Alpha context"))
    beta = store.insert_unit(_unit("beta", "Beta", "Beta context"))
    store.create_collection(
        "agent-handoff",
        description="Curated handoff context",
        metadata={"owner": "agents"},
    )
    store.add_unit_to_collection("agent-handoff", alpha.id)
    store.add_unit_to_collection("agent-handoff", beta.id)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "graph_collection_members",
            {"collection_name": "agent-handoff"},
        )
    )
    payload = json.loads(response[0].text)

    assert set(payload) == {"collection", "members"}
    assert payload["collection"]["name"] == "agent-handoff"
    assert payload["collection"]["description"] == "Curated handoff context"
    assert payload["collection"]["metadata"] == {"owner": "agents"}
    assert payload["collection"]["unit_count"] == 2
    assert {member["id"] for member in payload["members"]} == {alpha.id, beta.id}
    assert {
        "id",
        "title",
        "source_project",
        "source_id",
        "source_entity_type",
        "content_type",
        "tags",
        "created_at",
        "updated_at",
        "added_at",
    } <= set(payload["members"][0])


def test_graph_collection_members_honors_limit(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    alpha = store.insert_unit(_unit("alpha", "Alpha", "Alpha context"))
    beta = store.insert_unit(_unit("beta", "Beta", "Beta context"))
    store.create_collection("agent-handoff")
    store.add_unit_to_collection("agent-handoff", alpha.id)
    store.add_unit_to_collection("agent-handoff", beta.id)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "graph_collection_members",
            {"collection_name": "agent-handoff", "limit": 1},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["collection"]["unit_count"] == 2
    assert len(payload["members"]) == 1
    assert payload["members"][0]["id"] in {alpha.id, beta.id}


def test_graph_collection_members_validates_limit(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    Store(str(db_path)).close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "graph_collection_members",
            {"collection_name": "agent-handoff", "limit": 0},
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "error": "invalid_limit",
        "message": "limit must be a positive integer",
        "collection_name": "agent-handoff",
        "limit": 0,
    }


def test_graph_collection_members_returns_structured_error_for_missing_collection(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    Store(str(db_path)).close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "graph_collection_members",
            {"collection_name": "missing"},
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "collection": "missing",
        "members": [],
        "error": "collection_not_found",
        "message": "Collection not found: missing",
    }
