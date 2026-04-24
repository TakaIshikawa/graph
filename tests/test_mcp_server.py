"""Tests for the MCP server tools."""

from __future__ import annotations

import asyncio
import json

import networkx as nx

from graph.adapters.base import IngestResult
from graph.mcp import server as mcp_server
from graph.rag.embeddings import serialize_embedding
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class FakeAdapter:
    def __init__(self, name: str, entity_types: list[str], result: IngestResult | None = None):
        self.name = name
        self.entity_types = entity_types
        self.result = result or IngestResult()

    def ingest(self, *, since=None, entity_types=None):
        return self.result


def _populate_tags_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="solar-storage",
            source_entity_type="insight",
            title="Solar storage",
            content="Solar storage insight",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar", "storage"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="solar-grid",
            source_entity_type="knowledge_node",
            title="Solar grid",
            content="Solar grid finding",
            content_type=ContentType.FINDING,
            tags=["energy", "solar", "grid"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="battery-storage",
            source_entity_type="insight",
            title="Battery storage",
            content="Battery storage insight",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage", "battery"],
        ),
    ]:
        store.insert_unit(unit)


def _populate_duplicates_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="same-title-a",
            source_entity_type="insight",
            title="Repeated Import",
            content="First independent note about sales workflow cleanup.",
            content_type=ContentType.INSIGHT,
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="same-title-b",
            source_entity_type="insight",
            title=" repeated   import ",
            content="Second independent note about onboarding research.",
            content_type=ContentType.INSIGHT,
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="same-content-a",
            source_entity_type="insight",
            title="Storage market signal",
            content="Solar storage adoption is accelerating across midmarket operations teams this quarter.",
            content_type=ContentType.INSIGHT,
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="same-content-b",
            source_entity_type="insight",
            title="Battery adoption memo",
            content="Solar storage adoption is accelerating across midmarket operations teams this quarter rapidly.",
            content_type=ContentType.INSIGHT,
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="filtered-title",
            source_entity_type="knowledge_node",
            title="Repeated Import",
            content="Outside project duplicate title.",
            content_type=ContentType.FINDING,
        ),
    ]:
        store.insert_unit(unit)


def _populate_backlinks_graph(store: Store) -> tuple[str, str, str]:
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First node",
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="b",
            source_entity_type="knowledge_node",
            title="Node B",
            content="Second node",
        )
    )
    c = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="c",
            source_entity_type="insight",
            title="Node C",
            content="Third node",
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=a.id,
            to_unit_id=b.id,
            relation=EdgeRelation.BUILDS_ON,
            source=EdgeSource.MANUAL,
            metadata={"why": "reference"},
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=b.id,
            to_unit_id=c.id,
            relation=EdgeRelation.INSPIRES,
        )
    )
    return a.id, b.id, c.id


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
    entity_types = {
        "forty_two": ["knowledge_node"],
        "max": ["insight", "buildable_unit", "design_brief"],
        "presence": ["knowledge_item", "generated_content"],
        "me": ["profile"],
        "kindle": ["book", "highlight"],
        "sota": ["paper"],
    }
    monkeypatch.setattr(
        mcp_server,
        "_get_adapter",
        lambda name: FakeAdapter(name, entity_types[name]),
    )

    tools = asyncio.run(mcp_server.list_tools())
    ingest_tool = next(tool for tool in tools if tool.name == "ingest")
    assert "kindle" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "sota" in ingest_tool.inputSchema["properties"]["project"]["enum"]

    search_tool = next(tool for tool in tools if tool.name == "search")
    assert "kindle" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "sota" in search_tool.inputSchema["properties"]["source_project"]["enum"]

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
    sota_missing = next(
        item
        for item in statuses
        if item["source_project"] == "sota" and item["source_entity_type"] == "paper"
    )
    assert sota_missing == {
        "source_project": "sota",
        "source_entity_type": "paper",
        "has_sync_state": False,
        "last_sync_at": None,
        "last_source_id": None,
        "items_synced": 0,
    }


def test_ingest_all_includes_sota_and_search_can_filter_sota(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"

    sota_result = IngestResult()
    sota_result.units.append(
        KnowledgeUnit(
            source_project=SourceProject.SOTA,
            source_id="paper_2501.00001",
            source_entity_type="paper",
            title="SOTA Transformer Paper",
            content="Transformer routing breakthrough for retrieval systems",
            content_type=ContentType.FINDING,
        )
    )

    calls = []

    def fake_get_adapter(name: str):
        calls.append(name)
        if name == "sota":
            return FakeAdapter(name, ["paper"], sota_result)
        return FakeAdapter(name, ["placeholder"])

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )
    monkeypatch.setattr(mcp_server, "_get_adapter", fake_get_adapter)

    response = asyncio.run(
        mcp_server.call_tool("ingest", {"project": "all", "full": True})
    )
    payload = json.loads(response[0].text)

    assert calls == ["forty_two", "max", "presence", "me", "kindle", "sota"]
    assert payload == {"units_inserted": 1, "units_skipped": 0, "edges_inserted": 0}

    store = Store(str(db_path))
    try:
        state = store.get_sync_state("sota", "paper")
        assert state is not None
        assert state.items_synced == 1
    finally:
        store.close()

    response = asyncio.run(
        mcp_server.call_tool(
            "search",
            {
                "query": "Transformer",
                "source_project": "sota",
                "mode": "fulltext",
            },
        )
    )
    results = json.loads(response[0].text)
    assert [result["source_project"] for result in results] == ["sota"]
    assert results[0]["title"] == "SOTA Transformer Paper"


def test_saved_query_tools_create_list_run_and_delete(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    keep = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="approved",
            source_entity_type="insight",
            title="Solar approved insight",
            content="Solar energy storage market growth",
            content_type=ContentType.INSIGHT,
            metadata={"review_state": "approved"},
            tags=["energy", "solar"],
        )
    )
    skip = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="rejected",
            source_entity_type="insight",
            title="Solar rejected insight",
            content="Solar energy storage market risk",
            content_type=ContentType.INSIGHT,
            metadata={"review_state": "rejected"},
            tags=["energy", "solar"],
        )
    )
    store.fts_index_unit(keep)
    store.fts_index_unit(skip)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert {"save_query", "list_queries", "run_query", "delete_query"} <= tool_names

    save_response = asyncio.run(
        mcp_server.call_tool(
            "save_query",
            {
                "name": "approved-solar",
                "query": "solar",
                "mode": "fulltext",
                "limit": 5,
                "filters": {
                    "source_project": "max",
                    "content_type": "insight",
                    "tag": "energy",
                    "review_state": "approved",
                },
            },
        )
    )
    saved = json.loads(save_response[0].text)
    assert saved["name"] == "approved-solar"
    assert saved["filters"]["review_state"] == "approved"

    list_response = asyncio.run(mcp_server.call_tool("list_queries", {}))
    listed = json.loads(list_response[0].text)
    assert [item["name"] for item in listed["queries"]] == ["approved-solar"]

    run_response = asyncio.run(
        mcp_server.call_tool("run_query", {"name": "approved-solar"})
    )
    payload = json.loads(run_response[0].text)
    assert payload["saved_query"] == "approved-solar"
    assert payload["query"] == "solar"
    assert [result["title"] for result in payload["results"]] == [
        "Solar approved insight"
    ]

    delete_response = asyncio.run(
        mcp_server.call_tool("delete_query", {"name": "approved-solar"})
    )
    assert json.loads(delete_response[0].text) == {
        "name": "approved-solar",
        "deleted": True,
    }

    missing_delete = asyncio.run(
        mcp_server.call_tool("delete_query", {"name": "approved-solar"})
    )
    assert json.loads(missing_delete[0].text) == {
        "name": "approved-solar",
        "deleted": False,
        "error": "Saved query not found: approved-solar",
    }

    missing_run = asyncio.run(
        mcp_server.call_tool("run_query", {"name": "approved-solar"})
    )
    assert json.loads(missing_run[0].text) == {
        "name": "approved-solar",
        "found": False,
        "error": "Saved query not found: approved-solar",
    }


def test_backlinks_tool_returns_expanded_json_filters_and_missing_error(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _, b_id, _ = _populate_backlinks_graph(store)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "backlinks")
    assert tool.inputSchema["properties"]["direction"]["enum"] == [
        "incoming",
        "outgoing",
        "both",
    ]

    response = asyncio.run(mcp_server.call_tool("backlinks", {"unit_id": b_id}))
    payload = json.loads(response[0].text)
    assert payload["center"]["title"] == "Node B"
    assert {
        (link["direction"], link["relation"], link["unit"]["title"])
        for link in payload["links"]
    } == {
        ("incoming", "builds_on", "Node A"),
        ("outgoing", "inspires", "Node C"),
    }
    assert any(link["edge"]["metadata"] for link in payload["links"])

    filtered = asyncio.run(
        mcp_server.call_tool(
            "backlinks",
            {
                "unit_id": b_id,
                "direction": "outgoing",
                "relation": "inspires",
            },
        )
    )
    filtered_payload = json.loads(filtered[0].text)
    assert [link["unit"]["title"] for link in filtered_payload["links"]] == ["Node C"]

    missing = asyncio.run(mcp_server.call_tool("backlinks", {"unit_id": "missing"}))
    assert json.loads(missing[0].text)["error"] == "unit_not_found"


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


def test_export_obsidian_tool_uses_configured_vault_default(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First node",
        )
    )
    store.close()

    vault_path = tmp_path / "configured-vault"
    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )
    monkeypatch.setattr(mcp_server.settings, "obsidian_vault_path", str(vault_path))

    tools = asyncio.run(mcp_server.list_tools())
    export_tool = next(tool for tool in tools if tool.name == "export_obsidian")
    assert export_tool.inputSchema["properties"]["vault_path"]["default"] == str(vault_path)

    response = asyncio.run(
        mcp_server.call_tool(
            "export_obsidian",
            {
                "folder": "Graph",
                "clean": True,
            },
        )
    )

    assert json.loads(response[0].text) == {"notes_written": 1}
    assert (vault_path / "Graph" / "forty_two" / "Node A.md").exists()


def test_json_backup_tools_export_and_import_idempotently(tmp_path, monkeypatch):
    source_db = tmp_path / "source.db"
    target_db = tmp_path / "target.db"
    backup_path = tmp_path / "backup.json"

    store = Store(str(source_db))
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First searchable node",
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="b",
            source_entity_type="insight",
            title="Node B",
            content="Second searchable node",
            content_type=ContentType.INSIGHT,
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

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "export_json" for tool in tools)
    assert any(tool.name == "import_json" for tool in tools)

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(source_db)))
    export_response = asyncio.run(
        mcp_server.call_tool("export_json", {"path": str(backup_path)})
    )
    export_payload = json.loads(export_response[0].text)

    assert export_payload["units_exported"] == 2
    assert export_payload["edges_exported"] == 1
    assert backup_path.exists()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(target_db)))
    import_response = asyncio.run(
        mcp_server.call_tool("import_json", {"path": str(backup_path)})
    )
    import_payload = json.loads(import_response[0].text)

    assert import_payload["units_inserted"] == 2
    assert import_payload["edges_inserted"] == 1

    second_response = asyncio.run(
        mcp_server.call_tool("import_json", {"path": str(backup_path)})
    )
    second_payload = json.loads(second_response[0].text)

    assert second_payload["units_inserted"] == 0
    assert second_payload["units_updated"] == 2
    assert second_payload["edges_inserted"] == 0
    assert second_payload["edges_skipped"] == 1

    target = Store(str(target_db))
    try:
        assert target.count_units() == 2
        assert len(target.get_all_edges()) == 1
        assert target.fts_search("searchable")
    finally:
        target.close()


def test_export_graphml_tool_returns_path_and_counts(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    graphml_path = tmp_path / "graph.graphml"

    store = Store(str(db_path))
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First searchable node",
            tags=["energy", "solar"],
            utility_score=0.9,
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="b",
            source_entity_type="insight",
            title="Node B",
            content="Second searchable node",
            content_type=ContentType.INSIGHT,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=a.id,
            to_unit_id=b.id,
            relation=EdgeRelation.BUILDS_ON,
            weight=0.8,
        )
    )
    store.close()

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "export_graphml" for tool in tools)

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    response = asyncio.run(
        mcp_server.call_tool("export_graphml", {"path": str(graphml_path)})
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "path": str(graphml_path),
        "node_count": 2,
        "edge_count": 1,
    }
    assert graphml_path.exists()

    exported = nx.read_graphml(graphml_path)
    assert exported.nodes[a.id]["tags"] == "energy,solar"
    assert exported.nodes[a.id]["utility_score"] == 0.9
    assert exported.get_edge_data(a.id, b.id)["relation"] == "builds_on"
    assert exported.get_edge_data(a.id, b.id)["weight"] == 0.8
    assert exported.get_edge_data(a.id, b.id)["source"] == "inferred"


def test_analyze_tags_tool_returns_summary_and_detail_json(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_tags_graph(store)
    store.close()

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )

    tools = asyncio.run(mcp_server.list_tools())
    analyze_tool = next(tool for tool in tools if tool.name == "analyze_tags")
    assert "source_project" in analyze_tool.inputSchema["properties"]
    assert "content_type" in analyze_tool.inputSchema["properties"]

    summary_response = asyncio.run(
        mcp_server.call_tool("analyze_tags", {"limit": 2})
    )
    summary = json.loads(summary_response[0].text)
    assert [item["tag"] for item in summary["tags"]] == ["energy", "solar"]

    detail_response = asyncio.run(
        mcp_server.call_tool(
            "analyze_tags",
            {
                "tag": "energy",
                "source_project": "max",
                "content_type": "insight",
            },
        )
    )
    detail = json.loads(detail_response[0].text)
    assert detail["count"] == 2
    assert {unit["title"] for unit in detail["units"]} == {
        "Solar storage",
        "Battery storage",
    }
    assert [item["tag"] for item in detail["co_occurring_tags"]] == [
        "storage",
        "battery",
        "solar",
    ]


def test_analyze_duplicates_tool_returns_reasons_units_and_filters(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_duplicates_graph(store)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    analyze_tool = next(tool for tool in tools if tool.name == "analyze_duplicates")
    assert "source_project" in analyze_tool.inputSchema["properties"]
    assert "content_type" in analyze_tool.inputSchema["properties"]

    response = asyncio.run(
        mcp_server.call_tool(
            "analyze_duplicates",
            {
                "source_project": "max",
                "content_type": "insight",
            },
        )
    )
    payload = json.loads(response[0].text)
    by_reason = {item["reason"]: item for item in payload["results"]}
    assert set(by_reason) == {"same_title", "similar_content"}
    assert {unit["source_id"] for unit in by_reason["same_title"]["units"]} == {
        "same-title-a",
        "same-title-b",
    }
    assert {unit["source_id"] for unit in by_reason["similar_content"]["units"]} == {
        "same-content-a",
        "same-content-b",
    }
    assert all(
        unit["source_project"] == "max"
        for item in payload["results"]
        for unit in item["units"]
    )


def test_infer_edges_tool_returns_counts_and_inserts_edges(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="edge-a",
            source_entity_type="insight",
            title="Edge A",
            content="Edge A content",
            content_type=ContentType.INSIGHT,
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="edge-b",
            source_entity_type="insight",
            title="Edge B",
            content="Edge B content",
            content_type=ContentType.INSIGHT,
        )
    )
    c = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="edge-c",
            source_entity_type="knowledge_node",
            title="Edge C",
            content="Edge C content",
            content_type=ContentType.INSIGHT,
        )
    )
    store.update_embedding(a.id, serialize_embedding([1.0, 0.0]))
    store.update_embedding(b.id, serialize_embedding([0.9, 0.1]))
    store.update_embedding(c.id, serialize_embedding([1.0, 0.0]))
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "infer_edges" for tool in tools)

    dry_response = asyncio.run(
        mcp_server.call_tool(
            "infer_edges",
            {
                "source_project": "max",
                "content_type": "insight",
                "min_similarity": 0.8,
                "dry_run": True,
            },
        )
    )
    dry_payload = json.loads(dry_response[0].text)
    assert dry_payload["inserted"] == 0
    assert dry_payload["skipped"] == 0
    assert len(dry_payload["candidates"]) == 1

    normal_response = asyncio.run(
        mcp_server.call_tool(
            "infer_edges",
            {
                "source_project": "max",
                "content_type": "insight",
                "min_similarity": 0.8,
            },
        )
    )
    payload = json.loads(normal_response[0].text)
    assert payload["inserted"] == 1
    assert payload["skipped"] == 0

    store = Store(str(db_path))
    try:
        edges = store.get_all_edges()
        assert len(edges) == 1
        assert edges[0].relation == EdgeRelation.RELATES_TO
        assert edges[0].source == EdgeSource.INFERRED
        assert {edges[0].from_unit_id, edges[0].to_unit_id} == {a.id, b.id}
    finally:
        store.close()
