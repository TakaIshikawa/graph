from __future__ import annotations

import asyncio
import json

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _unit(source_id: str, title: str, content: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=SourceProject.MAX,
        source_id=source_id,
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=["mcp", "context-pack"],
    )


def test_export_context_pack_tool_returns_markdown_and_counts(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    alpha = store.insert_unit(_unit("alpha", "Alpha", "Alpha context"))
    beta = store.insert_unit(_unit("beta", "Beta", "Beta context"))
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=alpha.id,
            to_unit_id=beta.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "export_context_pack",
            {"unit_ids": [beta.id, alpha.id], "options": {"title": "Agent Handoff"}},
        )
    )
    payload = json.loads(response[0].text)

    assert set(payload) == {"markdown", "metadata"}
    assert payload["markdown"].startswith('---\ntitle: "Agent Handoff"\n')
    assert payload["markdown"].index("1. [Beta]") < payload["markdown"].index("2. [Alpha]")
    assert "### Beta" in payload["markdown"]
    assert "### Alpha" in payload["markdown"]
    assert "`" + alpha.id + "` --`relates_to`--> `" + beta.id + "`" in payload["markdown"]
    assert payload["metadata"]["requested_count"] == 2
    assert payload["metadata"]["exported_count"] == 2
    assert payload["metadata"]["requested_unit_ids"] == [beta.id, alpha.id]
    assert payload["metadata"]["exported_unit_ids"] == [beta.id, alpha.id]
    assert "path" not in payload["metadata"]


def test_export_context_pack_tool_is_registered_with_selector_schema():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "export_context_pack")

    assert "required" not in tool.inputSchema
    assert tool.inputSchema["properties"]["unit_ids"]["items"] == {"type": "string"}
    assert tool.inputSchema["properties"]["query"]["type"] == "string"
    assert tool.inputSchema["properties"]["tag"]["type"] == "string"
    assert tool.inputSchema["properties"]["mode"]["enum"] == ["fulltext", "semantic", "hybrid"]
    assert tool.inputSchema["properties"]["limit"]["minimum"] == 1


def test_export_context_pack_tool_selects_units_by_search_and_filters(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    alpha = store.insert_unit(_unit("alpha", "Alpha", "Alpha context"))
    beta = store.insert_unit(_unit("beta", "Beta", "Beta context"))
    other = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="other",
            source_entity_type="insight",
            title="Alpha Archive",
            content="Alpha archived context",
            content_type=ContentType.INSIGHT,
            tags=["archive"],
        )
    )
    store.fts_index_unit(alpha)
    store.fts_index_unit(beta)
    store.fts_index_unit(other)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "export_context_pack",
            {
                "query": "Alpha",
                "mode": "fulltext",
                "tag": "context-pack",
                "limit": 5,
                "title": "Filtered Pack",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["metadata"]["selector"] == "query"
    assert payload["metadata"]["requested_unit_ids"] == [alpha.id]
    assert payload["metadata"]["filters"] == {"tag": "context-pack"}
    assert "### Alpha" in payload["markdown"]
    assert "### Beta" not in payload["markdown"]
    assert "### Alpha Archive" not in payload["markdown"]


def test_export_context_pack_tool_selects_units_by_filters_without_query(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    alpha = store.insert_unit(_unit("alpha", "Alpha", "Alpha context"))
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="other",
            source_entity_type="insight",
            title="Other",
            content="Other context",
            content_type=ContentType.INSIGHT,
            tags=["archive"],
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "export_context_pack",
            {"tag": "context-pack", "limit": 1, "sort": "created_at_desc"},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["metadata"]["selector"] == "filters"
    assert payload["metadata"]["requested_unit_ids"] == [alpha.id]
    assert "### Alpha" in payload["markdown"]
    assert "### Other" not in payload["markdown"]


def test_export_context_pack_tool_returns_structured_error_for_unknown_ids(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    unit = store.insert_unit(_unit("known", "Known", "Known context"))
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "export_context_pack",
            {"unit_ids": [unit.id, "missing-unit"]},
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "error": "unknown_unit_ids",
        "message": "One or more requested unit ids were not found",
        "missing_unit_ids": ["missing-unit"],
        "metadata": {
            "requested_count": 2,
            "exported_count": 0,
            "found_count": 1,
        },
    }


def test_export_context_pack_tool_validates_unit_ids(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("export_context_pack", {"unit_ids": []}))
    payload = json.loads(response[0].text)

    assert payload["error"] == "invalid_unit_ids"
    assert payload["metadata"] == {"requested_count": 0, "exported_count": 0}
