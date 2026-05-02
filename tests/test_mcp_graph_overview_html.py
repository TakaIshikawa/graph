from __future__ import annotations

import asyncio
import json

from graph.mcp import server as mcp_server
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _unit(source_id: str, title: str, *, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=SourceProject.MAX,
        source_id=source_id,
        source_entity_type="insight",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def test_export_graph_overview_html_tool_is_listed_with_schema():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "export_graph_overview_html")

    assert tool.inputSchema["type"] == "object"
    assert "path" in tool.inputSchema["properties"]
    assert tool.inputSchema["properties"]["limit"]["minimum"] == 1


def test_export_graph_overview_html_writes_report_and_returns_metadata(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    output_path = tmp_path / "reports" / "overview.html"

    store = Store(str(db_path))
    a = store.insert_unit(_unit("a", "Solar storage", tags=["solar", "storage"]))
    b = store.insert_unit(_unit("b", "Grid planning", tags=["grid", "solar"]))
    store.insert_unit(_unit("c", "Isolated note", tags=[]))
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=a.id,
            to_unit_id=b.id,
            relation=EdgeRelation.BUILDS_ON,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    response = asyncio.run(
        mcp_server.call_tool(
            "export_graph_overview_html",
            {"path": str(output_path), "limit": 2},
        )
    )
    payload = json.loads(response[0].text)
    html = output_path.read_text(encoding="utf-8")

    assert payload == {
        "path": str(output_path),
        "exported": True,
        "graph": {
            "unit_count": 3,
            "edge_count": 1,
            "component_count": 2,
            "isolated_count": 1,
        },
        "report": {
            "top_tags": 2,
            "top_sources": 1,
            "central_units": 2,
            "components": 2,
            "warnings": 1,
        },
    }
    assert html.startswith("<!doctype html>")
    assert "<h1>Graph Overview</h1>" in html
    assert "Solar storage" in html
    assert "1 isolated units" in html


def test_export_graph_overview_html_defaults_to_current_directory(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    store.insert_unit(_unit("a", "Default path unit", tags=["default"]))
    store.close()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    response = asyncio.run(mcp_server.call_tool("export_graph_overview_html", {}))
    payload = json.loads(response[0].text)

    assert payload["path"] == "graph-overview.html"
    assert payload["exported"] is True
    assert (tmp_path / "graph-overview.html").exists()


def test_export_graph_overview_html_returns_clear_error_for_directory_path(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    store.insert_unit(_unit("a", "Directory path unit", tags=[]))
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    response = asyncio.run(
        mcp_server.call_tool("export_graph_overview_html", {"path": str(tmp_path)})
    )
    payload = json.loads(response[0].text)

    assert payload["path"] == str(tmp_path)
    assert payload["exported"] is False
    assert payload["error"] == "invalid_output_path"
    assert "directory" in payload["message"]
