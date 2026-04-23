"""Tests for the MCP server tools."""

from __future__ import annotations

import asyncio
import json

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


def test_sync_status_tool_lists_supported_pairs_and_handles_missing_state(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    store.upsert_sync_state(
        SyncState(
            source_project="max",
            source_entity_type="insight",
            last_sync_at="2026-04-24T00:00:00+00:00",
            last_source_id="ins-1",
            items_synced=7,
        )
    )
    store.close()

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )

    tools = asyncio.run(mcp_server.list_tools())
    ingest_tool = next(tool for tool in tools if tool.name == "ingest")
    assert "kindle" in ingest_tool.inputSchema["properties"]["project"]["enum"]

    search_tool = next(tool for tool in tools if tool.name == "search")
    assert "kindle" in search_tool.inputSchema["properties"]["source_project"]["enum"]

    assert any(tool.name == "sync_status" for tool in tools)

    response = asyncio.run(mcp_server.call_tool("sync_status", {}))
    payload = json.loads(response[0].text)
    statuses = payload["sync_states"]

    max_insight = next(
        item
        for item in statuses
        if item["source_project"] == "max" and item["source_entity_type"] == "insight"
    )
    assert max_insight == {
        "source_project": "max",
        "source_entity_type": "insight",
        "has_sync_state": True,
        "last_sync_at": "2026-04-24T00:00:00+00:00",
        "last_source_id": "ins-1",
        "items_synced": 7,
    }

    missing = next(
        item
        for item in statuses
        if item["source_project"] == "presence"
        and item["source_entity_type"] == "generated_content"
    )
    assert missing == {
        "source_project": "presence",
        "source_entity_type": "generated_content",
        "has_sync_state": False,
        "last_sync_at": None,
        "last_source_id": None,
        "items_synced": 0,
    }
    kindle_missing = next(
        item
        for item in statuses
        if item["source_project"] == "kindle" and item["source_entity_type"] == "book"
    )
    assert kindle_missing == {
        "source_project": "kindle",
        "source_entity_type": "book",
        "has_sync_state": False,
        "last_sync_at": None,
        "last_source_id": None,
        "items_synced": 0,
    }

def test_export_obsidian_tool_exports_same_vault_structure_as_cli(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First node",
            utility_score=0.9,
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="b",
            source_entity_type="insight",
            title="Node B",
            content="Second node",
            content_type=ContentType.INSIGHT,
            utility_score=0.8,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=a.id,
            to_unit_id=b.id,
            relation=EdgeRelation.BUILDS_ON,
        )
    )
    store.close()

    vault_path = tmp_path / "vault"
    export_folder = vault_path / "Graph"
    export_folder.mkdir(parents=True)
    (export_folder / "stale.txt").write_text("old content")

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "export_obsidian" for tool in tools)

    response = asyncio.run(
        mcp_server.call_tool(
            "export_obsidian",
            {
                "vault_path": str(vault_path),
                "folder": "Graph",
                "clean": True,
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {"notes_written": 2}
    assert not (export_folder / "stale.txt").exists()

    node_a_path = export_folder / "forty_two" / "Node A.md"
    node_b_path = export_folder / "max" / "Node B.md"
    index_path = export_folder / "Index.md"

    assert node_a_path.exists()
    assert node_b_path.exists()
    assert index_path.exists()

    node_a = node_a_path.read_text()
    node_b = node_b_path.read_text()
    index = index_path.read_text()

    assert "id: " in node_a
    assert "## Connections" in node_a
    assert "[[Graph/max/Node B|Node B]]" in node_a
    assert "## Referenced by" in node_b
    assert "[[Graph/forty_two/Node A|Node A]]" in node_b
    assert "# Knowledge Graph Index" in index
    assert "## forty_two (1)" in index
    assert "## max (1)" in index
