"""Tests for the MCP server tools."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import networkx as nx
import yaml

from graph.adapters.base import IngestResult
from graph.graph.service import GraphService
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


class StaticEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


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


def _populate_timeline_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="jan-solar",
            source_entity_type="insight",
            title="January solar",
            content="Solar storage",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar"],
            created_at=datetime.fromisoformat("2026-01-15T10:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="feb-grid",
            source_entity_type="knowledge_node",
            title="February grid",
            content="Grid finding",
            content_type=ContentType.FINDING,
            tags=["energy", "grid"],
            created_at=datetime.fromisoformat("2026-02-05T10:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="feb-battery",
            source_entity_type="insight",
            title="February battery",
            content="Battery storage",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage"],
            created_at=datetime.fromisoformat("2026-02-20T10:00:00+00:00"),
        ),
    ]:
        store.insert_unit(unit)


def _populate_tag_synonym_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="agent-hyphen",
            source_entity_type="insight",
            title="Agent hyphen",
            content="Agent note",
            content_type=ContentType.INSIGHT,
            tags=["ai-agent"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="agent-underscore",
            source_entity_type="insight",
            title="Agent underscore",
            content="Agent note",
            content_type=ContentType.INSIGHT,
            tags=["ai_agent"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="agent-plural",
            source_entity_type="knowledge_node",
            title="Agent plural",
            content="Agent note",
            content_type=ContentType.FINDING,
            tags=["AI Agents"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="unrelated",
            source_entity_type="knowledge_item",
            title="Unrelated",
            content="Unrelated note",
            content_type=ContentType.ARTIFACT,
            tags=["storage"],
        ),
    ]:
        store.insert_unit(unit)


def _populate_duplicates_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="canonical-a",
            source_entity_type="insight",
            title="Canonical A",
            content="Canonical duplicate A.",
            content_type=ContentType.INSIGHT,
            metadata={"canonical_url": "https://example.com/shared"},
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="canonical-b",
            source_entity_type="insight",
            title="Canonical B",
            content="Canonical duplicate B.",
            content_type=ContentType.INSIGHT,
            metadata={"canonical_url": "https://example.com/shared"},
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="title-a",
            source_entity_type="insight",
            title="Solar storage roadmap",
            content="Title duplicate A.",
            content_type=ContentType.INSIGHT,
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="title-b",
            source_entity_type="insight",
            title="Solar storage road map",
            content="Title duplicate B.",
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


def _populate_review_queue_graph(store: Store) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    old_isolated = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="old-isolated",
            source_entity_type="insight",
            title="Old isolated",
            content="Old unreviewed insight",
            content_type=ContentType.INSIGHT,
            utility_score=0.6,
            created_at=now - timedelta(days=420),
        )
    )
    recent_reviewed = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="recent-reviewed",
            source_entity_type="insight",
            title="Recent reviewed",
            content="Recent reviewed insight",
            content_type=ContentType.INSIGHT,
            metadata={"reviewed_at": now.isoformat()},
            utility_score=1.0,
            created_at=now - timedelta(days=2),
        )
    )
    neighbor = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="neighbor",
            source_entity_type="knowledge_node",
            title="Neighbor",
            content="Connected node",
            content_type=ContentType.FINDING,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=recent_reviewed.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="old-finding",
            source_entity_type="knowledge_node",
            title="Old finding",
            content="Old finding",
            content_type=ContentType.FINDING,
            created_at=now - timedelta(days=365),
        )
    )
    return old_isolated.id, recent_reviewed.id


def _populate_links_graph(store: Store) -> None:
    for unit in [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="markdown-link",
            source_entity_type="insight",
            title="Markdown citation",
            content="Read [docs](https://Example.com/docs).",
            content_type=ContentType.INSIGHT,
            metadata={"source_url": "https://meta.test/report)."},
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="duplicate-link",
            source_entity_type="knowledge_node",
            title="Repeated citation",
            content="Repeated https://example.com/docs",
            content_type=ContentType.FINDING,
        ),
    ]:
        store.insert_unit(unit)


def _populate_edge_suggestion_graph(store: Store) -> tuple[str, str]:
    solar = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="solar-storage",
            source_entity_type="insight",
            title="Solar storage roadmap",
            content="Battery grid planning references https://example.com/docs.",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar", "storage"],
        )
    )
    battery = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="battery-storage",
            source_entity_type="insight",
            title="Battery storage plan",
            content="Grid battery storage notes cite https://example.com/docs.",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage", "battery"],
        )
    )
    grid = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="solar-grid",
            source_entity_type="knowledge_node",
            title="Solar grid plan",
            content="Solar storage grid reference https://example.com/docs.",
            content_type=ContentType.FINDING,
            tags=["energy", "solar", "grid"],
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=grid.id,
            to_unit_id=solar.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    return battery.id, solar.id


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
            weight=0.75,
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


def test_mcp_collection_tools_manage_members_with_unit_summaries(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-collection-1",
            source_entity_type="manual",
            title="MCP collection unit",
            content="Unit content",
        )
    )
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert "collection_create" in tool_names
    assert "collection_members" in tool_names
    assert "graph_collection_members" in tool_names

    created = asyncio.run(
        mcp_server.call_tool(
            "collection_create",
            {"name": "reading", "description": "Reading list"},
        )
    )
    added = asyncio.run(
        mcp_server.call_tool("collection_add", {"name": "reading", "unit_id": unit.id})
    )
    duplicate = asyncio.run(
        mcp_server.call_tool("collection_add", {"name": "reading", "unit_id": unit.id})
    )
    members = asyncio.run(mcp_server.call_tool("collection_members", {"name": "reading"}))
    deleted = asyncio.run(mcp_server.call_tool("collection_delete", {"name": "reading"}))

    created_payload = json.loads(created[0].text)
    added_payload = json.loads(added[0].text)
    duplicate_payload = json.loads(duplicate[0].text)
    members_payload = json.loads(members[0].text)
    deleted_payload = json.loads(deleted[0].text)

    assert created_payload["created"] is True
    assert added_payload["added"] is True
    assert duplicate_payload["already_member"] is True
    assert members_payload["members"][0]["id"] == unit.id
    assert members_payload["members"][0]["title"] == "MCP collection unit"
    assert deleted_payload["deleted"] is True

    verify = Store(str(db_path))
    try:
        assert verify.get_unit(unit.id) is not None
    finally:
        verify.close()


def test_mcp_collection_diff_returns_counts_summaries_and_errors(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    alpha = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="mcp-collection-diff-alpha",
            source_entity_type="manual",
            title="Alpha",
            content="Left only",
        )
    )
    beta = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="mcp-collection-diff-beta",
            source_entity_type="manual",
            title="Beta",
            content="Shared",
        )
    )
    gamma = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="mcp-collection-diff-gamma",
            source_entity_type="manual",
            title="Gamma",
            content="Right only",
        )
    )
    delta = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="mcp-collection-diff-delta",
            source_entity_type="manual",
            title="Delta",
            content="Right only limited out",
        )
    )
    store.create_collection("left")
    store.create_collection("right")
    store.add_unit_to_collection("left", alpha.id)
    store.add_unit_to_collection("left", beta.id)
    store.add_unit_to_collection("right", beta.id)
    store.add_unit_to_collection("right", gamma.id)
    store.add_unit_to_collection("right", delta.id)
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool_by_name = {tool.name: tool for tool in tools}
    assert "collection_diff" in tool_by_name
    diff_schema = tool_by_name["collection_diff"].inputSchema
    assert diff_schema["required"] == ["left_name", "right_name"]
    assert diff_schema["properties"]["include_units"] == {
        "type": "boolean",
        "default": True,
        "description": "Include unit summaries for left-only, right-only, and shared members",
    }

    response = asyncio.run(
        mcp_server.call_tool(
            "collection_diff",
            {"left_name": "left", "right_name": "right", "limit": 1},
        )
    )
    payload = json.loads(response[0].text)
    assert payload["counts"] == {
        "left": 2,
        "right": 3,
        "left_only": 1,
        "right_only": 2,
        "both": 1,
    }
    assert [unit["title"] for unit in payload["left_only"]] == ["Alpha"]
    assert [unit["title"] for unit in payload["right_only"]] == ["Delta"]
    assert [unit["title"] for unit in payload["both"]] == ["Beta"]
    assert set(payload["left_only"][0]) >= {
        "id",
        "title",
        "source_project",
        "source_id",
        "source_entity_type",
        "content_type",
        "tags",
    }

    counts_only = asyncio.run(
        mcp_server.call_tool(
            "collection_diff",
            {
                "left_name": "left",
                "right_name": "right",
                "include_units": False,
            },
        )
    )
    counts_only_payload = json.loads(counts_only[0].text)
    assert counts_only_payload["counts"] == payload["counts"]
    assert counts_only_payload["left_only"] == []
    assert counts_only_payload["right_only"] == []
    assert counts_only_payload["both"] == []

    missing = asyncio.run(
        mcp_server.call_tool(
            "collection_diff",
            {"left_name": "left", "right_name": "missing"},
        )
    )
    missing_payload = json.loads(missing[0].text)
    assert missing_payload == {
        "left_name": "left",
        "right_name": "missing",
        "error": "collection_not_found",
        "message": "Collection not found: missing",
    }


def test_mcp_collection_import_export_tools_and_schema(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    export_path = tmp_path / "collections.json"
    store = Store(str(db_path))
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="mcp-collection-export-1",
            source_entity_type="manual",
            title="MCP export unit",
            content="Unit content",
        )
    )
    store.create_collection("reading", description="Reading list")
    store.add_unit_to_collection("reading", unit.id)
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool_by_name = {tool.name: tool for tool in tools}
    assert {"export_collections", "import_collections"} <= set(tool_by_name)
    assert tool_by_name["import_collections"].inputSchema["properties"]["strict"] == {
        "type": "boolean",
        "default": False,
        "description": "Fail without writing memberships if any referenced unit is missing",
    }

    export_response = asyncio.run(
        mcp_server.call_tool("export_collections", {"path": str(export_path)})
    )
    export_payload = json.loads(export_response[0].text)
    assert export_payload["collections_exported"] == 1
    assert export_payload["memberships_exported"] == 1
    exported = json.loads(export_path.read_text())
    assert exported["schema_version"] == 1
    assert exported["collections"][0]["unit_ids"] == [unit.id]

    update_store = Store(str(db_path))
    update_store.delete_collection("reading")
    update_store.close()

    import_response = asyncio.run(
        mcp_server.call_tool("import_collections", {"path": str(export_path)})
    )
    import_payload = json.loads(import_response[0].text)
    assert import_payload["collections_inserted"] == 1
    assert import_payload["memberships_added"] == 1

    exported["collections"][0]["unit_ids"].append("missing-unit")
    export_path.write_text(json.dumps(exported))
    strict_response = asyncio.run(
        mcp_server.call_tool(
            "import_collections",
            {"path": str(export_path), "strict": True},
        )
    )
    strict_payload = json.loads(strict_response[0].text)
    assert strict_payload["error"] == "import_failed"
    assert "missing-unit" in strict_payload["message"]


def test_adapter_catalog_tool_lists_configuration_without_ingesting(tmp_path, monkeypatch):
    from graph.adapters import registry

    monkeypatch.setattr(mcp_server.settings, "csv_path", str(tmp_path / "items.csv"))
    monkeypatch.setattr(mcp_server.settings, "jsonl_path", "")
    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: (_ for _ in ()).throw(AssertionError("catalog should not open store")),
    )
    monkeypatch.setattr(registry, "list_adapters", lambda: ["csv", "jsonl", "broken"])

    def fake_get_adapter(name: str, **kwargs):
        if name == "broken":
            raise ImportError("missing optional package")
        return FakeAdapter(name, {"csv": ["csv_row"], "jsonl": ["jsonl_record"]}[name])

    monkeypatch.setattr(registry, "get_adapter", fake_get_adapter)

    tools = asyncio.run(mcp_server.list_tools())
    catalog_tool = next(tool for tool in tools if tool.name == "adapter_catalog")
    assert catalog_tool.inputSchema == {"type": "object", "properties": {}}

    response = asyncio.run(mcp_server.call_tool("adapter_catalog", {}))
    payload = json.loads(response[0].text)
    adapters = {adapter["name"]: adapter for adapter in payload["adapters"]}

    assert adapters["csv"] == {
        "name": "csv",
        "entity_types": ["csv_row"],
        "configured": True,
        "error": None,
    }
    assert adapters["jsonl"] == {
        "name": "jsonl",
        "entity_types": ["jsonl_record"],
        "configured": False,
        "error": None,
    }
    assert adapters["broken"]["entity_types"] == []
    assert adapters["broken"]["configured"] is False
    assert adapters["broken"]["error"] == "ImportError: missing optional package"


def test_sync_status_tool_lists_supported_pairs_and_handles_missing_state(tmp_path, monkeypatch):
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
        "feed": ["feed_item"],
        "bookmarks": ["bookmark"],
        "csv": ["csv_row"],
        "jsonl": ["jsonl_record"],
        "opml": ["outline"],
        "pdf": ["pdf_document"],
        "email": ["email_message"],
        "enex": ["note"],
        "text": ["text_document"],
        "html": ["html_document"],
        "ical": ["calendar_event"],
        "ipynb": ["notebook"],
        "ris": ["ris_record"],
        "git": ["git_commit"],
        "transcript": ["transcript"],
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
    assert "feed" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "bookmarks" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "csv" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "jsonl" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "opml" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "enex" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "text" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "html" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "ical" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "ipynb" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "ris" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "git" in ingest_tool.inputSchema["properties"]["project"]["enum"]
    assert "transcript" in ingest_tool.inputSchema["properties"]["project"]["enum"]

    search_tool = next(tool for tool in tools if tool.name == "search")
    assert "kindle" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "sota" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "feed" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "bookmarks" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "csv" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "jsonl" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "opml" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "enex" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "text" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "html" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "ical" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "ipynb" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "ris" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "git" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert "transcript" in search_tool.inputSchema["properties"]["source_project"]["enum"]
    assert search_tool.inputSchema["properties"]["created_after"]["type"] == "string"
    assert search_tool.inputSchema["properties"]["min_utility"]["type"] == "number"
    assert search_tool.inputSchema["properties"]["min_confidence"]["type"] == "number"
    assert search_tool.inputSchema["properties"]["metadata_key"]["type"] == "string"
    assert search_tool.inputSchema["properties"]["metadata_value"]["type"] == "string"
    assert search_tool.inputSchema["properties"]["snippet_length"]["type"] == "integer"
    assert search_tool.inputSchema["properties"]["snippet_length"]["minimum"] == 1
    assert "created_at_desc" in search_tool.inputSchema["properties"]["sort"]["enum"]
    search_facets_tool = next(tool for tool in tools if tool.name == "search_facets")
    assert search_facets_tool.inputSchema["properties"]["tag"]["type"] == "string"
    assert search_facets_tool.inputSchema["properties"]["exclude_tag"]["type"] == "string"
    context_tool = next(tool for tool in tools if tool.name == "context_pack")
    assert context_tool.inputSchema["properties"]["metadata_key"]["type"] == "string"
    assert context_tool.inputSchema["properties"]["metadata_value"]["type"] == "string"
    assert search_facets_tool.inputSchema["properties"]["mode"]["enum"] == [
        "hybrid",
        "semantic",
        "fulltext",
    ]
    create_unit_tool = next(tool for tool in tools if tool.name == "create_unit")
    assert create_unit_tool.inputSchema["properties"]["metadata_json"]["type"] == "string"
    assert create_unit_tool.inputSchema["properties"]["source_project"]["default"] == "me"
    assert create_unit_tool.inputSchema["properties"]["source_entity_type"]["default"] == "manual"
    save_query_tool = next(tool for tool in tools if tool.name == "save_query")
    assert save_query_tool.inputSchema["properties"]["created_before"]["type"] == "string"
    assert save_query_tool.inputSchema["properties"]["max_utility"]["type"] == "number"
    assert "utility_desc" in save_query_tool.inputSchema["properties"]["sort"]["enum"]
    assert save_query_tool.inputSchema["properties"]["schedule"]["enum"] == [
        "daily",
        "weekly",
        "monthly",
    ]
    run_query_tool = next(tool for tool in tools if tool.name == "run_query")
    assert run_query_tool.inputSchema["properties"]["limit"]["type"] == "integer"
    assert run_query_tool.inputSchema["properties"]["limit"]["minimum"] == 1

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
    feed_missing = next(
        item
        for item in statuses
        if item["source_project"] == "feed" and item["source_entity_type"] == "feed_item"
    )
    assert feed_missing == {
        "source_project": "feed",
        "source_entity_type": "feed_item",
        "has_sync_state": False,
        "last_sync_at": None,
        "last_source_id": None,
        "items_synced": 0,
    }
    bookmarks_missing = next(
        item
        for item in statuses
        if item["source_project"] == "bookmarks" and item["source_entity_type"] == "bookmark"
    )
    assert bookmarks_missing == {
        "source_project": "bookmarks",
        "source_entity_type": "bookmark",
        "has_sync_state": False,
        "last_sync_at": None,
        "last_source_id": None,
        "items_synced": 0,
    }
    csv_missing = next(
        item
        for item in statuses
        if item["source_project"] == "csv" and item["source_entity_type"] == "csv_row"
    )
    assert csv_missing == {
        "source_project": "csv",
        "source_entity_type": "csv_row",
        "has_sync_state": False,
        "last_sync_at": None,
        "last_source_id": None,
        "items_synced": 0,
    }
    jsonl_missing = next(
        item
        for item in statuses
        if item["source_project"] == "jsonl" and item["source_entity_type"] == "jsonl_record"
    )
    assert jsonl_missing == {
        "source_project": "jsonl",
        "source_entity_type": "jsonl_record",
        "has_sync_state": False,
        "last_sync_at": None,
        "last_source_id": None,
        "items_synced": 0,
    }


def test_stats_tool_returns_snapshot_payload(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First node",
            tags=["shared"],
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
            tags=["shared", "solar"],
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=a.id,
            to_unit_id=b.id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.MANUAL,
        )
    )
    store.update_embedding(a.id, serialize_embedding([1.0, 0.0]))
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    assert next(tool for tool in tools if tool.name == "stats")

    response = asyncio.run(mcp_server.call_tool("stats", {}))
    payload = json.loads(response[0].text)

    assert set(payload) == {
        "unit_counts",
        "edge_counts",
        "embedding_counts",
        "isolated_count",
        "top_degree_units",
    }
    assert payload["unit_counts"]["total"] == 2
    assert payload["unit_counts"]["by_tag"] == {"shared": 2, "solar": 1}
    assert payload["edge_counts"] == {
        "total": 1,
        "by_relation": {"relates_to": 1},
        "by_source": {"manual": 1},
    }
    assert payload["embedding_counts"] == {"with_embeddings": 1, "without_embeddings": 1}
    assert payload["isolated_count"] == 0
    assert [item["degree"] for item in payload["top_degree_units"]] == [1, 1]


def test_freshness_report_tool_returns_counts_and_stale_status(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    now = datetime.now(timezone.utc)
    store = Store(str(db_path))
    recent = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="mcp-fresh-recent",
            source_entity_type="insight",
            title="Recent freshness unit",
            content="Freshness content",
        )
    )
    old = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="mcp-fresh-old",
            source_entity_type="insight",
            title="Old freshness unit",
            content="Old freshness content",
        )
    )
    store.conn.execute(
        "UPDATE knowledge_units SET ingested_at = ? WHERE id = ?",
        ((now - timedelta(days=3)).isoformat(), recent.id),
    )
    store.conn.execute(
        "UPDATE knowledge_units SET ingested_at = ? WHERE id = ?",
        ((now - timedelta(days=8)).isoformat(), old.id),
    )
    store.conn.commit()
    store.upsert_sync_state(
        SyncState(
            source_project="max",
            source_entity_type="insight",
            last_sync_at=now - timedelta(days=8),
            items_synced=2,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    monkeypatch.setattr(
        mcp_server,
        "_get_adapter",
        lambda name: FakeAdapter(name, {"max": ["insight"], "presence": ["knowledge_item"]}[name]),
    )
    monkeypatch.setattr(mcp_server, "SUPPORTED_SYNC_PROJECTS", ["max", "presence"])

    tools = asyncio.run(mcp_server.list_tools())
    freshness_tool = next(tool for tool in tools if tool.name == "freshness_report")
    assert freshness_tool.inputSchema["properties"]["days"]["default"] == 7

    seven = asyncio.run(mcp_server.call_tool("freshness_report", {"days": 7}))
    seven_payload = json.loads(seven[0].text)
    by_target = {
        (item["source_project"], item["source_entity_type"]): item
        for item in seven_payload["results"]
    }
    assert seven_payload["days"] == 7
    assert by_target[("max", "insight")]["recent_unit_count"] == 1
    assert by_target[("max", "insight")]["total_unit_count"] == 2
    assert by_target[("max", "insight")]["last_sync_at"] is not None
    assert by_target[("max", "insight")]["age_days"] >= 8
    assert by_target[("max", "insight")]["stale"] is True
    assert by_target[("presence", "knowledge_item")] == {
        "source_project": "presence",
        "source_entity_type": "knowledge_item",
        "last_sync_at": None,
        "age_days": None,
        "recent_unit_count": 0,
        "total_unit_count": 0,
        "stale": True,
    }

    ten = asyncio.run(mcp_server.call_tool("freshness_report", {"days": 10}))
    ten_payload = json.loads(ten[0].text)
    ten_max = ten_payload["results"][0]
    assert ten_payload["days"] == 10
    assert ten_max["recent_unit_count"] == 2
    assert ten_max["stale"] is False


def test_embedding_status_tool_returns_grouped_report_and_stale_units(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    fresh = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="mcp-embed-fresh",
            source_entity_type="insight",
            title="MCP Fresh",
            content="Fresh content",
            content_type=ContentType.INSIGHT,
        )
    )
    stale = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="mcp-embed-stale",
            source_entity_type="insight",
            title="MCP Stale",
            content="Stale content",
            content_type=ContentType.INSIGHT,
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="mcp-embed-missing",
            source_entity_type="knowledge_node",
            title="MCP Missing",
            content="Missing content",
            content_type=ContentType.FINDING,
        )
    )
    store.update_embedding(fresh.id, serialize_embedding([1.0, 0.0]))
    store.update_embedding(stale.id, serialize_embedding([1.0, 0.0]))
    store.update_unit_fields(stale.id, content="Updated stale content")
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    embedding_tool = next(tool for tool in tools if tool.name == "embedding_status")
    assert "source_project" in embedding_tool.inputSchema["properties"]
    assert "content_type" in embedding_tool.inputSchema["properties"]
    assert embedding_tool.inputSchema["properties"]["show_stale"]["default"] == 0

    response = asyncio.run(
        mcp_server.call_tool(
            "embedding_status",
            {"source_project": "max", "show_stale": 5},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["filters"] == {"source_project": "max", "content_type": None}
    assert payload["totals"] == {
        "total": 2,
        "missing": 0,
        "fresh": 1,
        "stale": 1,
        "percent_fresh": 50.0,
    }
    assert payload["by_source_project"][0]["source_project"] == "max"
    assert payload["groups"][0]["content_type"] == "insight"
    assert payload["stale_units"] == [
        {
            "id": stale.id,
            "title": "MCP Stale",
            "source_project": "max",
            "content_type": "insight",
            "reason": "stale_embedding",
            "updated_at": payload["stale_units"][0]["updated_at"],
            "embedding_updated_at": payload["stale_units"][0]["embedding_updated_at"],
        }
    ]


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

    response = asyncio.run(mcp_server.call_tool("ingest", {"project": "all", "full": True}))
    payload = json.loads(response[0].text)

    assert calls == [
        "forty_two",
        "max",
        "presence",
        "me",
        "kindle",
        "sota",
        "feed",
        "bookmarks",
        "csv",
        "jsonl",
        "opml",
        "pdf",
        "email",
        "enex",
        "text",
        "html",
        "ical",
        "ipynb",
        "ris",
        "git",
        "transcript",
    ]
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


def test_ingest_tool_syncs_configured_feed_fixture(tmp_path, monkeypatch):
    feed = tmp_path / "feed.xml"
    feed.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>MCP Feed</title>
          <entry>
            <id>tag:example.com,2026:mcp-feed-1</id>
            <title>MCP local feed item</title>
            <link href="https://example.com/mcp-feed"/>
            <updated>2026-04-24T12:30:00Z</updated>
            <category term="mcp"/>
            <summary>Read through the MCP ingest tool.</summary>
          </entry>
        </feed>
        """,
        encoding="utf-8",
    )
    db_path = tmp_path / "graph.db"

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    monkeypatch.setattr(mcp_server.settings, "feed_sources", str(feed))

    response = asyncio.run(
        mcp_server.call_tool("ingest", {"project": "feed", "full": True})
    )
    payload = json.loads(response[0].text)

    assert payload == {"units_inserted": 1, "units_skipped": 0, "edges_inserted": 0}

    store = Store(str(db_path))
    try:
        unit = store.conn.execute(
            """SELECT * FROM knowledge_units
               WHERE source_project = 'feed' AND source_entity_type = 'feed_item'"""
        ).fetchone()
        assert unit is not None
        assert unit["title"] == "MCP local feed item"
        assert json.loads(unit["tags"]) == ["mcp"]
        metadata = json.loads(unit["metadata"])
        assert metadata["id"] == "tag:example.com,2026:mcp-feed-1"
        assert metadata["link"] == "https://example.com/mcp-feed"

        state = store.get_sync_state("feed", "feed_item")
        assert state is not None
        assert state.items_synced == 1
    finally:
        store.close()


def test_ingest_tool_syncs_configured_ical_fixture(tmp_path, monkeypatch):
    calendar = tmp_path / "calendar.ics"
    calendar.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:mcp-event-1
SUMMARY:MCP Calendar Event
DESCRIPTION:Read through the MCP ingest tool.
DTSTART:20260424T123000Z
CATEGORIES:mcp
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )
    db_path = tmp_path / "graph.db"

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    monkeypatch.setattr(mcp_server.settings, "ical_path", str(calendar))

    response = asyncio.run(mcp_server.call_tool("ingest", {"project": "ical", "full": True}))
    payload = json.loads(response[0].text)

    assert payload == {"units_inserted": 1, "units_skipped": 0, "edges_inserted": 0}

    store = Store(str(db_path))
    try:
        unit = store.get_unit_by_source(
            "calendar",
            "calendar.ics#mcp-event-1",
            "calendar_event",
        )
        assert unit is not None
        assert unit.title == "MCP Calendar Event"
        assert unit.tags == ["mcp"]
        assert unit.metadata["uid"] == "mcp-event-1"

        state = store.get_sync_state("ical", "calendar_event")
        assert state is not None
        assert state.items_synced == 1
    finally:
        store.close()


def test_ingest_tool_syncs_configured_ipynb_fixture(tmp_path, monkeypatch):
    root = tmp_path / "notebooks"
    root.mkdir()
    notebook = root / "mcp.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "metadata": {
                    "title": "MCP Notebook",
                    "tags": ["mcp", "notebook"],
                    "kernelspec": {"language": "python", "name": "python3"},
                    "language_info": {"name": "python"},
                },
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": "Read through the MCP ingest tool.",
                    },
                    {"cell_type": "code", "source": ["value = 42\n", "value\n"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "graph.db"

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    monkeypatch.setattr(mcp_server.settings, "ipynb_root", str(root))

    response = asyncio.run(
        mcp_server.call_tool("ingest", {"project": "ipynb", "full": True})
    )
    payload = json.loads(response[0].text)

    assert payload == {"units_inserted": 1, "units_skipped": 0, "edges_inserted": 0}

    store = Store(str(db_path))
    try:
        unit = store.get_unit_by_source("me", "mcp.ipynb", "notebook")
        assert unit is not None
        assert unit.title == "MCP Notebook"
        assert unit.tags == ["mcp", "notebook"]
        assert "Read through the MCP ingest tool." in unit.content
        assert unit.metadata["code_cell_count"] == 1

        state = store.get_sync_state("ipynb", "notebook")
        assert state is not None
        assert state.items_synced == 1
    finally:
        store.close()


def test_extract_references_tool_returns_candidates_and_inserts_edges(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    target = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="reference-target",
            source_entity_type="insight",
            title="Reference Target",
            content="Target content",
            content_type=ContentType.ARTIFACT,
            metadata={"source_url": "https://example.com/reference-target"},
        )
    )
    source = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="reference-source",
            source_entity_type="insight",
            title="Reference Source",
            content="Mentions https://example.com/reference-target.",
            content_type=ContentType.INSIGHT,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    extract_tool = next(tool for tool in tools if tool.name == "extract_references")
    assert extract_tool.inputSchema["properties"]["dry_run"]["default"] is False
    assert "source_project" in extract_tool.inputSchema["properties"]
    assert "content_type" in extract_tool.inputSchema["properties"]

    dry_response = asyncio.run(
        mcp_server.call_tool(
            "extract_references",
            {
                "source_project": "max",
                "content_type": "insight",
                "dry_run": True,
            },
        )
    )
    dry_payload = json.loads(dry_response[0].text)
    assert dry_payload["inserted"] == 0
    assert dry_payload["would_insert"] == 1
    assert dry_payload["candidates"][0]["status"] == "would_insert"
    assert dry_payload["candidates"][0]["to_unit_id"] == target.id

    normal_response = asyncio.run(
        mcp_server.call_tool(
            "extract_references",
            {
                "source_project": "max",
                "content_type": "insight",
            },
        )
    )
    payload = json.loads(normal_response[0].text)
    assert payload["inserted"] == 1
    assert payload["candidates"][0]["from_unit_id"] == source.id

    store = Store(str(db_path))
    try:
        edges = store.get_all_edges()
        assert len(edges) == 1
        assert edges[0].relation == EdgeRelation.REFERENCES
        assert edges[0].source == EdgeSource.INFERRED
        assert edges[0].from_unit_id == source.id
        assert edges[0].to_unit_id == target.id
    finally:
        store.close()


def test_similar_units_tool_returns_payload_and_excludes_seed(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    seed = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="seed",
            source_entity_type="insight",
            title="Solar storage seed",
            content="Solar storage adoption",
            content_type=ContentType.INSIGHT,
            tags=["energy"],
        )
    )
    match = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="match",
            source_entity_type="insight",
            title="Solar storage match",
            content="Solar storage demand",
            content_type=ContentType.INSIGHT,
            tags=["energy"],
        )
    )
    store.update_embedding(seed.id, serialize_embedding([1.0, 0.0]))
    store.update_embedding(match.id, serialize_embedding([0.9, 0.1]))
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "similar_units" for tool in tools)

    response = asyncio.run(
        mcp_server.call_tool(
            "similar_units",
            {
                "unit_id": seed.id,
                "limit": 5,
                "source_project": "max",
                "content_type": "insight",
                "tag": "energy",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["seed_id"] == seed.id
    assert payload["source_mode"] == "embedding"
    assert [result["id"] for result in payload["results"]] == [match.id]
    assert payload["results"][0]["reason"] == "embedding_similarity"
    assert payload["results"][0]["source_mode"] == "embedding"


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
            utility_score=0.92,
            created_at=datetime.fromisoformat("2026-04-22T00:00:00+00:00"),
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
            utility_score=0.4,
            created_at=datetime.fromisoformat("2026-04-23T00:00:00+00:00"),
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
                "sort": "created_at_desc",
                "schedule": "weekly",
                "created_before": "2026-04-23T00:00:00+00:00",
                "max_utility": 0.93,
                "filters": {
                    "source_project": "max",
                    "content_type": "insight",
                    "tag": "energy",
                    "review_state": "approved",
                    "created_after": "2026-04-21T00:00:00+00:00",
                    "min_utility": 0.9,
                },
            },
        )
    )
    saved = json.loads(save_response[0].text)
    assert saved["name"] == "approved-solar"
    assert saved["schedule"] == "weekly"
    assert saved["last_run_at"] is None
    assert saved["filters"]["review_state"] == "approved"
    assert saved["filters"]["created_after"] == "2026-04-21T00:00:00+00:00"
    assert saved["filters"]["created_before"] == "2026-04-23T00:00:00+00:00"
    assert saved["filters"]["min_utility"] == 0.9
    assert saved["filters"]["max_utility"] == 0.93
    assert saved["filters"]["sort"] == "created_at_desc"

    list_response = asyncio.run(mcp_server.call_tool("list_queries", {}))
    listed = json.loads(list_response[0].text)
    assert [item["name"] for item in listed["queries"]] == ["approved-solar"]

    run_response = asyncio.run(mcp_server.call_tool("run_query", {"name": "approved-solar"}))
    payload = json.loads(run_response[0].text)
    assert payload["saved_query"] == "approved-solar"
    assert payload["query"] == "solar"
    assert payload["sort"] == "created_at_desc"
    assert payload["schedule"] == "weekly"
    assert payload["last_run_at"] is not None
    assert datetime.fromisoformat(payload["last_run_at"]).tzinfo is not None
    assert payload["filters"] == saved["filters"]
    assert payload["saved_query_metadata"]["name"] == "approved-solar"
    assert payload["saved_query_metadata"]["limit"] == 5
    assert payload["saved_query_metadata"]["effective_limit"] == 5
    assert payload["saved_query_metadata"]["filters"] == saved["filters"]
    assert [result["title"] for result in payload["results"]] == ["Solar approved insight"]
    assert payload["result_summaries"] == [
        {
            key: payload["results"][0][key]
            for key in [
                "id",
                "title",
                "source_project",
                "source_id",
                "source_entity_type",
                "content_type",
                "tags",
                "snippet",
                "created_at",
                "updated_at",
                "utility_score",
            ]
        }
    ]

    search_response = asyncio.run(
        mcp_server.call_tool(
            "search",
            {
                "query": "solar",
                "mode": "fulltext",
                "created_after": "2026-04-21T00:00:00+00:00",
                "created_before": "2026-04-23T00:00:00+00:00",
                "min_utility": 0.9,
                "max_utility": 0.93,
                "sort": "created_at_desc",
            },
        )
    )
    search_payload = json.loads(search_response[0].text)
    assert search_payload["sort"] == "created_at_desc"
    assert [result["title"] for result in search_payload["results"]] == [
        "Solar approved insight"
    ]
    assert search_payload["results"][0]["snippet"]

    delete_response = asyncio.run(mcp_server.call_tool("delete_query", {"name": "approved-solar"}))
    assert json.loads(delete_response[0].text) == {
        "name": "approved-solar",
        "deleted": True,
    }

    missing_delete = asyncio.run(mcp_server.call_tool("delete_query", {"name": "approved-solar"}))
    assert json.loads(missing_delete[0].text) == {
        "name": "approved-solar",
        "deleted": False,
        "error": "Saved query not found: approved-solar",
    }

    missing_run = asyncio.run(mcp_server.call_tool("run_query", {"name": "approved-solar"}))
    assert json.loads(missing_run[0].text) == {
        "name": "approved-solar",
        "found": False,
        "error": "Saved query not found: approved-solar",
        "error_code": "saved_query_not_found",
    }


def test_run_saved_query_limit_override_caps_results_without_mutating_query(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    inserted = []
    for day, source_id in [(24, "newest"), (23, "middle"), (22, "oldest")]:
        inserted.append(
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=source_id,
                    source_entity_type="insight",
                    title=f"Solar {source_id}",
                    content="Solar storage planning",
                    content_type=ContentType.INSIGHT,
                    tags=["energy"],
                    created_at=datetime.fromisoformat(
                        f"2026-04-{day}T00:00:00+00:00"
                    ),
                )
            )
        )
    for unit in inserted:
        store.fts_index_unit(unit)
    saved = store.save_query(
        name="solar-review",
        query="solar",
        mode="fulltext",
        limit=3,
        filters={"source_project": "max", "sort": "created_at_desc"},
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool("run_query", {"name": "solar-review", "limit": 2})
    )
    payload = json.loads(response[0].text)

    assert payload["saved_query"] == "solar-review"
    assert payload["saved_query_metadata"]["limit"] == 3
    assert payload["saved_query_metadata"]["requested_limit"] == 2
    assert payload["saved_query_metadata"]["effective_limit"] == 2
    assert [result["title"] for result in payload["results"]] == [
        "Solar newest",
        "Solar middle",
    ]
    assert [summary["title"] for summary in payload["result_summaries"]] == [
        "Solar newest",
        "Solar middle",
    ]

    check_store = Store(str(db_path))
    try:
        unchanged = check_store.get_saved_query("solar-review")
    finally:
        check_store.close()
    assert unchanged["limit"] == saved["limit"]
    assert unchanged["filters"] == saved["filters"]


def test_saved_query_import_export_tools_round_trip(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    export_path = tmp_path / "queries.json"
    store = Store(str(db_path))
    store.save_query(
        name="approved-solar",
        query="solar",
        mode="hybrid",
        limit=5,
        filters={"source_project": "max", "review_state": "approved"},
        schedule="monthly",
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert {"export_queries", "import_queries"} <= tool_names

    export_response = asyncio.run(
        mcp_server.call_tool("export_queries", {"path": str(export_path)})
    )
    export_payload = json.loads(export_response[0].text)
    assert export_payload["exported"] == 1
    assert export_payload["schema_version"] == 2
    exported = json.loads(export_path.read_text())
    assert exported["queries"][0]["query"] == "solar"
    assert exported["queries"][0]["mode"] == "hybrid"
    assert exported["queries"][0]["limit"] == 5
    assert exported["queries"][0]["schedule"] == "monthly"
    assert exported["queries"][0]["last_run_at"] is None
    assert exported["queries"][0]["filters"] == {
        "source_project": "max",
        "review_state": "approved",
    }

    update_store = Store(str(db_path))
    update_store.save_query(
        name="approved-solar",
        query="outdated",
        mode="semantic",
        limit=1,
        filters={"tag": "old"},
    )
    update_store.close()

    import_response = asyncio.run(
        mcp_server.call_tool("import_queries", {"path": str(export_path)})
    )
    assert json.loads(import_response[0].text) == {
        "inserted": 0,
        "path": str(export_path),
        "skipped": 0,
        "updated": 1,
    }

    second_response = asyncio.run(
        mcp_server.call_tool("import_queries", {"path": str(export_path)})
    )
    assert json.loads(second_response[0].text)["skipped"] == 1

    invalid_path = tmp_path / "invalid-queries.json"
    invalid_path.write_text(json.dumps({"schema_version": 999, "queries": []}))
    invalid_response = asyncio.run(
        mcp_server.call_tool("import_queries", {"path": str(invalid_path)})
    )
    invalid_payload = json.loads(invalid_response[0].text)
    assert invalid_payload["error"] == "import_failed"
    assert "Unsupported saved queries schema_version 999" in invalid_payload["message"]

    legacy_path = tmp_path / "legacy-queries.json"
    legacy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queries": [
                    {
                        "name": "legacy",
                        "query": "stale",
                        "mode": "fulltext",
                        "limit": 10,
                        "filters": {},
                    }
                ],
            }
        )
    )
    legacy_response = asyncio.run(
        mcp_server.call_tool("import_queries", {"path": str(legacy_path)})
    )
    assert json.loads(legacy_response[0].text)["inserted"] == 1
    imported_store = Store(str(db_path))
    try:
        legacy = imported_store.get_saved_query("legacy")
        assert legacy is not None
        assert legacy["schedule"] is None
        assert legacy["last_run_at"] is None
    finally:
        imported_store.close()


def test_search_facets_tool_returns_parseable_json(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    units = [
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="approved",
            source_entity_type="insight",
            title="Solar approved insight",
            content="Solar energy storage market growth",
            content_type=ContentType.INSIGHT,
            metadata={"review_state": "approved"},
            tags=["energy", "solar"],
            utility_score=0.92,
            created_at=datetime.fromisoformat("2026-04-22T00:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="brief",
            source_entity_type="design_brief",
            title="Solar approved brief",
            content="Solar energy storage market brief",
            content_type=ContentType.DESIGN_BRIEF,
            metadata={"review_state": "approved"},
            tags=["energy", "solar"],
            utility_score=0.8,
            created_at=datetime.fromisoformat("2026-04-24T00:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="note",
            source_entity_type="knowledge_node",
            title="Solar forty two note",
            content="Solar energy storage market note",
            content_type=ContentType.FINDING,
            tags=["energy", "solar", "grid"],
            utility_score=0.95,
            created_at=datetime.fromisoformat("2026-04-21T00:00:00+00:00"),
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="archived",
            source_entity_type="insight",
            title="Solar archived insight",
            content="Solar archived energy storage market",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar", "archive"],
            utility_score=0.7,
            created_at=datetime.fromisoformat("2026-04-20T00:00:00+00:00"),
        ),
    ]
    for unit in units:
        inserted = store.insert_unit(unit)
        store.fts_index_unit(inserted)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "search_facets",
            {
                "query": "solar",
                "mode": "fulltext",
                "source_project": "max",
                "tag": "energy",
                "exclude_tag": "archive",
            },
        )
    )
    payload = json.loads(response[0].text)
    assert payload["query"] == "solar"
    assert payload["mode"] == "fulltext"
    assert payload["filters"] == {
        "exclude_tag": "archive",
        "source_project": "max",
        "tag": "energy",
    }
    assert payload["total_matches"] == 2
    assert payload["facets"]["source_project"] == {"max": 2}
    assert payload["facets"]["source_entity_type"] == {
        "design_brief": 1,
        "insight": 1,
    }
    assert payload["facets"]["content_type"] == {
        "design_brief": 1,
        "insight": 1,
    }
    assert payload["facets"]["tag"] == {"energy": 2, "solar": 2}


def test_search_tool_hybrid_results_include_scores_and_snippets(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="hybrid",
            source_entity_type="insight",
            title="Solar hybrid insight",
            content="Solar energy storage market growth",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar"],
        )
    )
    store.fts_index_unit(unit)
    store.update_embedding(unit.id, serialize_embedding([1.0, 0.0]))
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    monkeypatch.setattr(
        "graph.rag.embeddings.get_embedding_provider",
        lambda *args, **kwargs: StaticEmbeddingProvider(),
    )

    response = asyncio.run(
        mcp_server.call_tool(
            "search",
            {"query": "solar", "mode": "hybrid", "limit": 1},
        )
    )
    results = json.loads(response[0].text)
    assert len(results) == 1
    assert results[0]["title"] == "Solar hybrid insight"
    assert isinstance(results[0]["score"], float)
    assert results[0]["snippet"]


def test_search_tool_rejects_identical_tag_and_exclude_tag(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    Store(str(db_path)).close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "search",
            {"query": "solar", "tag": "archive", "exclude_tag": "archive"},
        )
    )

    payload = json.loads(response[0].text)
    assert payload["error"] == "tag and exclude_tag cannot be identical."


def test_search_tool_filters_by_metadata_path_and_validates_pair(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    keep = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="metadata-keep",
            source_entity_type="insight",
            title="Solar metadata keep",
            content="Solar metadata search",
            content_type=ContentType.INSIGHT,
            metadata={"project": {"area": "grid"}},
        )
    )
    skip = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="metadata-skip",
            source_entity_type="insight",
            title="Solar metadata skip",
            content="Solar metadata search",
            content_type=ContentType.INSIGHT,
            metadata={"project": {"area": "storage"}},
        )
    )
    store.fts_index_unit(keep)
    store.fts_index_unit(skip)
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "search",
            {
                "query": "solar",
                "metadata_key": "project.area",
                "metadata_value": "grid",
            },
        )
    )
    results = json.loads(response[0].text)
    assert [result["title"] for result in results] == ["Solar metadata keep"]

    error_response = asyncio.run(
        mcp_server.call_tool(
            "search",
            {"query": "solar", "metadata_key": "project.area"},
        )
    )
    payload = json.loads(error_response[0].text)
    assert payload["error"] == "metadata_key and metadata_value must be supplied together."


def test_search_tool_respects_snippet_length(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="snippet-length",
            source_entity_type="insight",
            title="Snippet length insight",
            content=("Opening context without the term. " * 6)
            + "Solar storage economics improve fast.",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar"],
        )
    )
    store.fts_index_unit(unit)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "search",
            {
                "query": "storage economics",
                "mode": "fulltext",
                "limit": 1,
                "snippet_length": 60,
            },
        )
    )
    results = json.loads(response[0].text)
    snippet = results[0]["snippet"]
    assert len(snippet) <= 60
    assert "storage economics" in snippet


def test_search_tool_invalid_sort_returns_clear_error(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "search",
            {"query": "solar", "mode": "fulltext", "sort": "newest"},
        )
    )
    payload = json.loads(response[0].text)
    assert "Unknown sort: newest" in payload["error"]
    assert "created_at_desc" in payload["valid_sorts"]


def test_export_search_html_tool_invocation_writes_report(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    export_path = tmp_path / "search.html"
    store = Store(str(db_path))
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="html-export",
            source_entity_type="insight",
            title="Solar MCP <unsafe>",
            content="Solar MCP export content",
            content_type=ContentType.INSIGHT,
            tags=["solar"],
            metadata={"review_state": "approved"},
        )
    )
    store.fts_index_unit(unit)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    tools = asyncio.run(mcp_server.list_tools())
    export_tool = next(tool for tool in tools if tool.name == "export_search_html")
    assert export_tool.inputSchema["properties"]["snippet_length"]["minimum"] == 1

    response = asyncio.run(
        mcp_server.call_tool(
            "export_search_html",
            {
                "path": str(export_path),
                "query": "solar",
                "mode": "fulltext",
                "source_project": "max",
                "review_state": "approved",
                "snippet_length": 40,
            },
        )
    )
    payload = json.loads(response[0].text)
    html = export_path.read_text(encoding="utf-8")

    assert payload["path"] == str(export_path)
    assert payload["result_count"] == 1
    assert payload["query"] == "solar"
    assert payload["mode"] == "fulltext"
    assert payload["filters"] == {"review_state": "approved", "source_project": "max"}
    assert "Solar MCP &lt;unsafe&gt;" in html
    assert "Solar MCP <unsafe>" not in html


def test_context_pack_tool_returns_filtered_results_and_graph_context(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    center = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="context-center",
            source_entity_type="insight",
            title="Solar context center",
            content="Solar context content " * 10,
            content_type=ContentType.INSIGHT,
            tags=["solar"],
            metadata={"project": {"area": "grid"}},
        )
    )
    filtered_out = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="context-other",
            source_entity_type="knowledge_node",
            title="Solar other project",
            content="Solar content from another project",
            content_type=ContentType.FINDING,
            tags=["solar"],
            metadata={"project": {"area": "storage"}},
        )
    )
    neighbor = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="context-neighbor",
            source_entity_type="insight",
            title="Neighbor insight",
            content="Neighbor content " * 10,
            content_type=ContentType.INSIGHT,
        )
    )
    archived = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="context-archive",
            source_entity_type="insight",
            title="Solar archived context",
            content="Solar archived context content",
            content_type=ContentType.INSIGHT,
            tags=["archive"],
        )
    )
    store.fts_index_unit(center)
    store.fts_index_unit(filtered_out)
    store.fts_index_unit(archived)
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=center.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.MANUAL,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "context_pack")
    assert "source_project" in tool.inputSchema["properties"]
    assert tool.inputSchema["properties"]["neighbor_depth"]["maximum"] == 2

    response = asyncio.run(
        mcp_server.call_tool(
            "context_pack",
            {
                "query": "solar",
                "mode": "fulltext",
                "source_project": "max",
                "exclude_tag": "archive",
                "metadata_key": "project.area",
                "metadata_value": "grid",
                "sort": "updated_at_desc",
                "limit": 5,
                "neighbor_depth": 2,
                "char_budget": 35,
            },
        )
    )
    payload = json.loads(response[0].text)
    assert payload["sort"] == "updated_at_desc"
    assert payload["metadata"]["sort"] == "updated_at_desc"
    assert [unit["id"] for unit in payload["ranked_units"]] == [center.id]
    assert payload["filters"] == {
        "exclude_tag": "archive",
        "metadata_key": "project.area",
        "metadata_value": "grid",
        "source_project": "max",
    }
    assert archived.id not in {unit["id"] for unit in payload["ranked_units"]}
    assert payload["neighbors"][0]["id"] == neighbor.id
    assert payload["selected_edges"][0]["relation"] == "relates_to"
    assert payload["metadata"]["content_chars_used"] <= 35


def test_update_and_delete_unit_tools_reindex_and_clean_edges(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    manual = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-1",
            source_entity_type="manual",
            title="Original solar note",
            content="Original solar content",
            metadata={"owner": "me", "review_state": "draft"},
            tags=["solar"],
        )
    )
    neighbor = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-2",
            source_entity_type="manual",
            title="Neighbor",
            content="Neighbor content",
        )
    )
    store.fts_index_unit(manual)
    store.fts_index_unit(neighbor)
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=manual.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert {"update_unit", "delete_unit"} <= tool_names

    update_response = asyncio.run(
        mcp_server.call_tool(
            "update_unit",
            {
                "unit_id": manual.id,
                "title": "Updated battery note",
                "content": "Updated battery content",
                "content_type": "finding",
                "tags": ["battery"],
                "metadata": {"review_state": "approved", "priority": "high"},
            },
        )
    )
    payload = json.loads(update_response[0].text)
    assert payload["updated"] is True
    assert payload["unit"]["title"] == "Updated battery note"
    assert payload["unit"]["metadata"] == {
        "owner": "me",
        "priority": "high",
        "review_state": "approved",
    }

    verify = Store(str(db_path))
    try:
        assert verify.fts_search("battery")[0]["unit_id"] == manual.id
        assert verify.fts_search("Original") == []
    finally:
        verify.close()

    delete_response = asyncio.run(mcp_server.call_tool("delete_unit", {"unit_id": manual.id}))
    assert json.loads(delete_response[0].text) == {
        "unit_id": manual.id,
        "deleted": True,
        "edges_deleted": 1,
    }

    verify = Store(str(db_path))
    try:
        assert verify.get_unit(manual.id) is None
        assert verify.get_all_edges() == []
        assert verify.fts_search("battery") == []
    finally:
        verify.close()


def test_unit_metadata_tools_set_remove_and_report_errors(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-metadata",
            source_entity_type="manual",
            title="Metadata target",
            content="Metadata content",
            metadata={"owner": "me", "review": {"state": "draft", "priority": "low"}},
        )
    )
    store.fts_index_unit(unit)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert {"unit_metadata_set", "unit_metadata_remove"} <= tool_names

    set_response = asyncio.run(
        mcp_server.call_tool(
            "unit_metadata_set",
            {"unit_id": unit.id, "path": "review.owner", "value": "alice"},
        )
    )
    set_payload = json.loads(set_response[0].text)
    assert set_payload["updated"] is True
    assert set_payload["unit"]["metadata"]["review"] == {
        "owner": "alice",
        "priority": "low",
        "state": "draft",
    }

    remove_response = asyncio.run(
        mcp_server.call_tool(
            "unit_metadata_remove",
            {"unit_id": unit.id, "path": "review.priority"},
        )
    )
    remove_payload = json.loads(remove_response[0].text)
    assert remove_payload["updated"] is True
    assert remove_payload["unit"]["metadata"] == {
        "owner": "me",
        "review": {"owner": "alice", "state": "draft"},
    }

    invalid_response = asyncio.run(
        mcp_server.call_tool(
            "unit_metadata_set",
            {"unit_id": unit.id, "path": "owner.name", "value": "alice"},
        )
    )
    invalid_payload = json.loads(invalid_response[0].text)
    assert invalid_payload["error"] == "invalid_metadata_path"
    assert "non-object value" in invalid_payload["message"]

    missing_response = asyncio.run(
        mcp_server.call_tool(
            "unit_metadata_remove",
            {"unit_id": "missing", "path": "review.owner"},
        )
    )
    assert json.loads(missing_response[0].text)["error"] == "unit_not_found"


def test_metadata_value_histogram_tool_returns_counts_and_honors_limit(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    for source_id, state in [
        ("approved-a", "approved"),
        ("approved-b", "approved"),
        ("draft-a", "draft"),
        ("archived-a", "archived"),
    ]:
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=source_id,
                source_entity_type="insight",
                title=f"Review {source_id}",
                content="Review metadata note",
                content_type=ContentType.INSIGHT,
                metadata={"review": {"state": state}},
            )
        )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    histogram_tool = next(
        tool for tool in tools if tool.name == "metadata_value_histogram"
    )
    assert histogram_tool.inputSchema["required"] == ["metadata_key"]
    assert histogram_tool.inputSchema["properties"]["metadata_key"]["type"] == "string"
    assert histogram_tool.inputSchema["properties"]["limit"]["minimum"] == 1

    response = asyncio.run(
        mcp_server.call_tool(
            "metadata_value_histogram",
            {"metadata_key": "review.state"},
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "path": "review.state",
        "source_project": None,
        "unit_count": 4,
        "missing_count": 0,
        "value_count": 4,
        "values": [
            {"value": "approved", "value_type": "string", "count": 2},
            {"value": "archived", "value_type": "string", "count": 1},
            {"value": "draft", "value_type": "string", "count": 1},
        ],
    }

    limited_response = asyncio.run(
        mcp_server.call_tool(
            "metadata_value_histogram",
            {"metadata_key": "review.state", "limit": 1},
        )
    )
    limited_payload = json.loads(limited_response[0].text)
    assert limited_payload["values"] == [
        {"value": "approved", "value_type": "string", "count": 2}
    ]


def test_metadata_value_histogram_tool_returns_empty_values_for_unknown_key(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="known-metadata",
            source_entity_type="insight",
            title="Known metadata",
            content="Known metadata note",
            content_type=ContentType.INSIGHT,
            metadata={"review": {"state": "approved"}},
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "metadata_value_histogram",
            {"metadata_key": "review.missing"},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["path"] == "review.missing"
    assert payload["unit_count"] == 1
    assert payload["missing_count"] == 1
    assert payload["value_count"] == 0
    assert payload["values"] == []


def test_create_unit_tool_validates_metadata_and_indexes_unit(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "create_unit",
            {
                "title": "Manual battery note",
                "content": "Ad hoc sodium battery observation",
                "content_type": "finding",
                "tags": ["manual", "battery"],
                "metadata_json": '{"owner":"me"}',
                "confidence": 0.7,
                "utility_score": 8.5,
            },
        )
    )
    payload = json.loads(response[0].text)
    unit = payload["unit"]
    assert payload["created"] is True
    assert unit["id"]
    assert unit["source_project"] == "me"
    assert unit["source_entity_type"] == "manual"
    assert unit["source_id"]
    assert unit["metadata"] == {"owner": "me"}
    assert unit["confidence"] == 0.7
    assert unit["utility_score"] == 8.5

    verify = Store(str(db_path))
    try:
        assert verify.fts_search("sodium")[0]["unit_id"] == unit["id"]
    finally:
        verify.close()

    invalid_response = asyncio.run(
        mcp_server.call_tool(
            "create_unit",
            {
                "title": "Invalid metadata",
                "content": "Content",
                "metadata_json": "{not json",
            },
        )
    )
    invalid_payload = json.loads(invalid_response[0].text)
    assert invalid_payload == {
        "created": False,
        "error": "--metadata-json must be a valid JSON object.",
    }


def test_merge_units_tool_dry_run_and_apply(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    target = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="merge-target",
            source_entity_type="manual",
            title="Target note",
            content="Target content",
            metadata={"owner": "target"},
            tags=["solar"],
        )
    )
    source = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="merge-source",
            source_entity_type="manual",
            title="Source note",
            content="Source content",
            metadata={"owner": "source", "status": "approved"},
            tags=["battery"],
        )
    )
    neighbor = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="merge-neighbor",
            source_entity_type="insight",
            title="Neighbor",
            content="Neighbor content",
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=source.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "merge_units" for tool in tools)

    dry_response = asyncio.run(
        mcp_server.call_tool(
            "merge_units",
            {"source_id": source.id, "target_id": target.id, "dry_run": True},
        )
    )
    dry_payload = json.loads(dry_response[0].text)
    assert dry_payload["dry_run"] is True
    assert dry_payload["rewired_edge_counts"]["total"] == 1

    verify = Store(str(db_path))
    try:
        assert verify.get_unit(source.id) is not None
        assert len(verify.get_all_edges()) == 1
    finally:
        verify.close()

    merge_response = asyncio.run(
        mcp_server.call_tool("merge_units", {"source_id": source.id, "target_id": target.id})
    )
    payload = json.loads(merge_response[0].text)
    assert payload["merged"] is True
    assert payload["deleted_unit_id"] == source.id

    verify = Store(str(db_path))
    try:
        assert verify.get_unit(source.id) is None
        merged = verify.get_unit(target.id)
        assert merged.tags == ["solar", "battery"]
        assert merged.metadata["status"] == "approved"
        edges = verify.get_all_edges()
        assert len(edges) == 1
        assert edges[0].from_unit_id == target.id
        assert edges[0].to_unit_id == neighbor.id
    finally:
        verify.close()

    missing = asyncio.run(
        mcp_server.call_tool("merge_units", {"source_id": "missing", "target_id": target.id})
    )
    assert json.loads(missing[0].text)["error"] == "unit_not_found"


def test_pin_and_unpin_unit_tools_preserve_metadata_and_report_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-pin",
            source_entity_type="manual",
            title="Pinned solar reference",
            content="Evergreen solar content",
            metadata={"owner": "me", "review_state": "approved"},
            tags=["solar"],
        )
    )
    store.fts_index_unit(unit)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert {"pin_unit", "unpin_unit"} <= tool_names

    pin_response = asyncio.run(
        mcp_server.call_tool(
            "pin_unit",
            {"unit_id": unit.id, "reason": "evergreen"},
        )
    )
    pin_payload = json.loads(pin_response[0].text)
    assert pin_payload["updated"] is True
    assert pin_payload["unit"]["tags"] == ["solar"]
    assert pin_payload["unit"]["content"] == "Evergreen solar content"
    assert pin_payload["unit"]["metadata"]["owner"] == "me"
    assert pin_payload["unit"]["metadata"]["pinned"] is True
    assert pin_payload["unit"]["metadata"]["pin_reason"] == "evergreen"
    assert pin_payload["unit"]["metadata"]["pinned_at"]

    unpin_response = asyncio.run(mcp_server.call_tool("unpin_unit", {"unit_id": unit.id}))
    unpin_payload = json.loads(unpin_response[0].text)
    assert unpin_payload["updated"] is True
    assert unpin_payload["unit"]["metadata"] == {
        "owner": "me",
        "review_state": "approved",
    }

    missing_response = asyncio.run(mcp_server.call_tool("pin_unit", {"unit_id": "missing"}))
    assert json.loads(missing_response[0].text) == {
        "unit_id": "missing",
        "updated": False,
        "error": "unit_not_found",
        "message": "Unit not found: missing",
    }


def test_pinned_units_tool_returns_same_structured_payload_with_filters(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    older = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="older-pin",
            source_entity_type="insight",
            title="Older pinned",
            content="Older pinned content",
            content_type=ContentType.INSIGHT,
            tags=["solar", "workspace"],
            metadata={
                "pinned": True,
                "pinned_at": "2026-01-01T00:00:00+00:00",
                "pin_reason": "older",
            },
        )
    )
    newer = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="newer-pin",
            source_entity_type="insight",
            title="Newer pinned",
            content="Newer pinned content",
            content_type=ContentType.INSIGHT,
            tags=["solar", "workspace"],
            metadata={
                "pinned": True,
                "pinned_at": "2026-01-02T00:00:00+00:00",
                "pin_reason": "newer",
            },
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="wrong-tag",
            source_entity_type="insight",
            title="Wrong tag pinned",
            content="Wrong tag pinned content",
            content_type=ContentType.INSIGHT,
            tags=["archive"],
            metadata={
                "pinned": True,
                "pinned_at": "2026-01-03T00:00:00+00:00",
            },
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    pinned_tool = next(tool for tool in tools if tool.name == "pinned_units")
    assert pinned_tool.inputSchema["properties"]["tag"]["type"] == "string"

    response = asyncio.run(
        mcp_server.call_tool(
            "pinned_units",
            {
                "source_project": "max",
                "content_type": "insight",
                "tag": "workspace",
                "limit": 2,
            },
        )
    )
    payload = json.loads(response[0].text)

    assert [unit["id"] for unit in payload["units"]] == [newer.id, older.id]
    assert payload["units"][0]["pin_reason"] == "newer"
    assert payload["units"][0]["pinned_at"] == "2026-01-02T00:00:00+00:00"
    assert "content" not in payload["units"][0]
    assert payload["filters"] == {
        "source_project": "max",
        "content_type": "insight",
        "tag": "workspace",
        "limit": 2,
    }

    with_content = asyncio.run(
        mcp_server.call_tool("pinned_units", {"tag": "workspace", "include_content": True})
    )
    content_payload = json.loads(with_content[0].text)
    assert content_payload["units"][0]["content"] == "Newer pinned content"


def test_integrity_audit_tool_reports_and_repairs_fts(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    unit = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="audit",
            source_entity_type="insight",
            title="Audit target",
            content="Search drift",
        )
    )
    store.conn.execute(
        "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
        ("deleted-unit", "Deleted", "stale content", "stale"),
    )
    store.conn.commit()
    store.close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    assert "integrity_audit" in {tool.name for tool in tools}

    response = asyncio.run(mcp_server.call_tool("integrity_audit", {}))
    payload = json.loads(response[0].text)
    assert payload["categories"]["units_missing_fts_rows"]["count"] == 1
    assert payload["categories"]["stale_fts_rows"]["count"] == 1

    repaired = asyncio.run(mcp_server.call_tool("integrity_audit", {"repair_fts": True}))
    repaired_payload = json.loads(repaired[0].text)
    assert repaired_payload["repair"]["requested"] is True
    assert repaired_payload["repair"]["fts_rows_inserted"] == 1
    assert repaired_payload["repair"]["fts_rows_deleted"] == 1
    assert repaired_payload["categories"]["units_missing_fts_rows"]["count"] == 0
    assert repaired_payload["categories"]["stale_fts_rows"]["count"] == 0

    verify = Store(str(db_path))
    try:
        assert verify.get_unit(unit.id) is not None
        assert verify.fts_search("Search")[0]["unit_id"] == unit.id
    finally:
        verify.close()


def test_edge_management_tools_list_update_delete_and_report_errors(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    center = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-1",
            source_entity_type="manual",
            title="Center",
            content="Center content",
        )
    )
    neighbor = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-2",
            source_entity_type="manual",
            title="Neighbor",
            content="Neighbor content",
        )
    )
    edge = store.insert_edge(
        KnowledgeEdge(
            from_unit_id=center.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.INFERRED,
            metadata={"old": "value"},
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool_names = {tool.name for tool in tools}
    assert {"list_edges", "update_edge", "delete_edge"} <= tool_names

    listed = asyncio.run(mcp_server.call_tool("list_edges", {"unit_id": center.id}))
    list_payload = json.loads(listed[0].text)
    assert list_payload["edges"][0]["id"] == edge.id
    assert list_payload["edges"][0]["direction"] == "outgoing"
    assert list_payload["edges"][0]["relation"] == "relates_to"
    assert list_payload["edges"][0]["source"] == "inferred"
    assert list_payload["edges"][0]["metadata"] == {"old": "value"}
    assert list_payload["edges"][0]["neighbor"]["title"] == "Neighbor"

    updated = asyncio.run(
        mcp_server.call_tool(
            "update_edge",
            {
                "edge_id": edge.id,
                "relation": "inspires",
                "weight": 0.25,
                "source": "manual",
                "metadata": {"new": "value"},
            },
        )
    )
    update_payload = json.loads(updated[0].text)
    assert update_payload["updated"] is True
    assert update_payload["edge"]["relation"] == "inspires"
    assert update_payload["edge"]["weight"] == 0.25
    assert update_payload["edge"]["source"] == "manual"
    assert update_payload["edge"]["metadata"] == {"old": "value", "new": "value"}

    verify = Store(str(db_path))
    try:
        assert verify.get_backlinks(neighbor.id)["links"][0]["relation"] == "inspires"
    finally:
        verify.close()

    invalid = asyncio.run(
        mcp_server.call_tool(
            "update_edge",
            {"edge_id": edge.id, "relation": "invalid"},
        )
    )
    assert "invalid" in json.loads(invalid[0].text)["error"]

    deleted = asyncio.run(mcp_server.call_tool("delete_edge", {"edge_id": edge.id}))
    assert json.loads(deleted[0].text) == {"edge_id": edge.id, "deleted": True}

    missing = asyncio.run(mcp_server.call_tool("delete_edge", {"edge_id": edge.id}))
    missing_payload = json.loads(missing[0].text)
    assert missing_payload["deleted"] is False
    assert missing_payload["error"] == "edge_not_found"

    verify = Store(str(db_path))
    try:
        assert verify.get_unit(center.id) is not None
        assert verify.get_unit(neighbor.id) is not None
        assert verify.get_all_edges() == []
    finally:
        verify.close()


def test_rename_edge_relation_tool_dry_run_execute_no_match_and_errors(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    source = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="relation-tool-source",
            source_entity_type="manual",
            title="Source",
            content="Source content",
        )
    )
    target = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="relation-tool-target",
            source_entity_type="manual",
            title="Target",
            content="Target content",
        )
    )
    edge = store.insert_edge(
        KnowledgeEdge(
            from_unit_id=source.id,
            to_unit_id=target.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    rename_tool = next(tool for tool in tools if tool.name == "rename_edge_relation")
    assert rename_tool.inputSchema["properties"]["dry_run"]["default"] is False
    assert rename_tool.inputSchema["required"] == ["old_relation", "new_relation"]

    dry_run = asyncio.run(
        mcp_server.call_tool(
            "rename_edge_relation",
            {
                "old_relation": "relates_to",
                "new_relation": "references",
                "dry_run": True,
            },
        )
    )
    dry_payload = json.loads(dry_run[0].text)
    assert dry_payload["dry_run"] is True
    assert dry_payload["matched_count"] == 1
    assert dry_payload["updated_count"] == 1
    assert dry_payload["affected_edge_ids"] == [edge.id]
    assert dry_payload["sample_edge_ids"] == [edge.id]

    verify = Store(str(db_path))
    assert verify.get_edge(edge.id).relation == EdgeRelation.RELATES_TO  # type: ignore[union-attr]
    verify.close()

    executed = asyncio.run(
        mcp_server.call_tool(
            "rename_edge_relation",
            {
                "old_relation": "relates_to",
                "new_relation": "references",
            },
        )
    )
    execute_payload = json.loads(executed[0].text)
    assert execute_payload["dry_run"] is False
    assert execute_payload["matched_count"] == 1
    assert execute_payload["updated_count"] == 1

    verify = Store(str(db_path))
    assert verify.get_edge(edge.id).relation == EdgeRelation.REFERENCES  # type: ignore[union-attr]
    verify.close()

    no_match = asyncio.run(
        mcp_server.call_tool(
            "rename_edge_relation",
            {
                "old_relation": "relates_to",
                "new_relation": "references",
                "dry_run": True,
            },
        )
    )
    assert json.loads(no_match[0].text)["matched_count"] == 0

    invalid = asyncio.run(
        mcp_server.call_tool(
            "rename_edge_relation",
            {
                "old_relation": "references",
                "new_relation": "references",
            },
        )
    )
    invalid_payload = json.loads(invalid[0].text)
    assert invalid_payload["updated_count"] == 0
    assert "must be different" in invalid_payload["error"]


def test_delete_edges_bulk_tool_dry_run_confirm_and_filters(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    center = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-1",
            source_entity_type="manual",
            title="Center",
            content="Center content",
        )
    )
    neighbor = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-2",
            source_entity_type="manual",
            title="Neighbor",
            content="Neighbor content",
        )
    )
    other = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="max-1",
            source_entity_type="insight",
            title="Other",
            content="Other content",
        )
    )
    target = store.insert_edge(
        KnowledgeEdge(
            from_unit_id=center.id,
            to_unit_id=neighbor.id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.INFERRED,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=center.id,
            to_unit_id=other.id,
            relation=EdgeRelation.INSPIRES,
            source=EdgeSource.INFERRED,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "delete_edges_bulk" for tool in tools)

    dry_run = asyncio.run(
        mcp_server.call_tool(
            "delete_edges_bulk",
            {
                "relation": "relates_to",
                "source": "inferred",
                "source_project": "me",
            },
        )
    )
    dry_payload = json.loads(dry_run[0].text)
    assert dry_payload["dry_run"] is True
    assert dry_payload["matched_count"] == 1
    assert dry_payload["deleted_count"] == 0
    assert dry_payload["edges"][0]["id"] == target.id
    assert dry_payload["edges"][0]["to_unit"]["title"] == "Neighbor"

    blocked = asyncio.run(
        mcp_server.call_tool(
            "delete_edges_bulk",
            {"relation": "relates_to", "dry_run": False},
        )
    )
    assert json.loads(blocked[0].text)["error"] == "confirmation_required"

    deleted = asyncio.run(
        mcp_server.call_tool(
            "delete_edges_bulk",
            {
                "relation": "relates_to",
                "source": "inferred",
                "source_project": "me",
                "dry_run": False,
                "confirm": True,
            },
        )
    )
    delete_payload = json.loads(deleted[0].text)
    assert delete_payload["deleted_count"] == 1

    verify = Store(str(db_path))
    try:
        edges = verify.get_all_edges()
        assert len(edges) == 1
        assert edges[0].id != target.id
    finally:
        verify.close()


def test_import_edges_csv_tool_matches_cli_structured_payload(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    u1 = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-1",
            source_entity_type="manual",
            title="Unit 1",
            content="Content 1",
        )
    )
    u2 = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id="manual-2",
            source_entity_type="manual",
            title="Unit 2",
            content="Content 2",
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=u1.id,
            to_unit_id=u2.id,
            relation=EdgeRelation.RELATES_TO,
        )
    )
    store.close()
    csv_path = tmp_path / "edges.csv"
    csv_path.write_text(
        "\n".join(
            [
                "from_unit_id,to_unit_id,relation,weight,source,metadata_json",
                f'{u1.id},{u2.id},relates_to,1,manual,{{"note":"duplicate"}}',
                f'{u2.id},{u1.id},references,0.25,manual,{{"note":"valid"}}',
                f"missing,{u1.id},references,1,manual,{{}}",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    assert "import_edges_csv" in {tool.name for tool in tools}

    dry_run = asyncio.run(
        mcp_server.call_tool(
            "import_edges_csv",
            {"path": str(csv_path), "dry_run": True},
        )
    )
    dry_payload = json.loads(dry_run[0].text)
    assert dry_payload["dry_run"] is True
    assert dry_payload["inserted"] == 1
    assert dry_payload["skipped_existing"] == 1
    assert len(dry_payload["invalid"]) == 1

    verify = Store(str(db_path))
    try:
        assert len(verify.get_all_edges()) == 1
    finally:
        verify.close()

    applied = asyncio.run(mcp_server.call_tool("import_edges_csv", {"path": str(csv_path)}))
    apply_payload = json.loads(applied[0].text)
    assert apply_payload["inserted"] == 1
    assert apply_payload["skipped_existing"] == 1

    verify = Store(str(db_path))
    try:
        edges = verify.get_all_edges()
        assert len(edges) == 2
        imported = next(edge for edge in edges if edge.from_unit_id == u2.id)
        assert imported.relation == EdgeRelation.REFERENCES
        assert imported.weight == 0.25
        assert imported.metadata == {"note": "valid"}
    finally:
        verify.close()


def test_backlinks_tool_returns_expanded_json_filters_and_missing_error(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _, b_id, _ = _populate_backlinks_graph(store)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "backlinks")
    assert tool.inputSchema["properties"]["depth"]["minimum"] == 1
    assert tool.inputSchema["properties"]["depth"]["maximum"] == 3
    assert tool.inputSchema["properties"]["direction"]["enum"] == [
        "incoming",
        "outgoing",
        "both",
    ]
    assert "source_project" in tool.inputSchema["properties"]
    assert "content_type" in tool.inputSchema["properties"]
    assert "tag" in tool.inputSchema["properties"]

    response = asyncio.run(
        mcp_server.call_tool("backlinks", {"unit_id": b_id, "direction": "both"})
    )
    payload = json.loads(response[0].text)
    assert payload["center"]["title"] == "Node B"
    assert payload["filters"] == {
        "depth": 1,
        "direction": "both",
        "relation": None,
        "source_project": None,
        "content_type": None,
        "tag": None,
        "limit": 20,
    }
    assert [edge["source_unit"]["title"] for edge in payload["inbound_edges"]] == ["Node A"]
    assert [edge["target_unit"]["title"] for edge in payload["outbound_edges"]] == ["Node C"]
    assert {unit["title"] for unit in payload["neighboring_units"]} == {"Node A", "Node C"}
    assert {
        (link["direction"], link["relation"], link["unit"]["title"]) for link in payload["links"]
    } == {
        ("incoming", "builds_on", "Node A"),
        ("outgoing", "inspires", "Node C"),
    }
    assert any(link["edge"]["metadata"] for link in payload["links"])
    incoming = next(link for link in payload["links"] if link["direction"] == "incoming")
    assert incoming["source_unit"]["title"] == "Node A"
    assert incoming["target_unit"]["title"] == "Node B"

    filtered = asyncio.run(
        mcp_server.call_tool(
            "backlinks",
            {
                "unit_id": b_id,
                "direction": "outgoing",
                "relation": "inspires",
                "source_project": "max",
            },
        )
    )
    filtered_payload = json.loads(filtered[0].text)
    assert [link["unit"]["title"] for link in filtered_payload["links"]] == ["Node C"]
    assert filtered_payload["center"]["title"] == "Node B"
    assert filtered_payload["inbound_edges"] == []
    assert [edge["neighbor_unit"]["title"] for edge in filtered_payload["outbound_edges"]] == [
        "Node C"
    ]

    default_response = asyncio.run(mcp_server.call_tool("backlinks", {"unit_id": b_id}))
    default_payload = json.loads(default_response[0].text)
    assert len(default_payload["inbound_edges"]) == 1
    assert len(default_payload["outbound_edges"]) == 1

    missing = asyncio.run(mcp_server.call_tool("backlinks", {"unit_id": "missing"}))
    assert json.loads(missing[0].text)["error"] == "unit_not_found"

    missing_id = asyncio.run(mcp_server.call_tool("backlinks", {}))
    missing_id_payload = json.loads(missing_id[0].text)
    assert missing_id_payload["error"] == "missing_unit_id"
    assert missing_id_payload["center"] is None

    invalid_depth = asyncio.run(mcp_server.call_tool("backlinks", {"unit_id": b_id, "depth": 0}))
    invalid_depth_payload = json.loads(invalid_depth[0].text)
    assert invalid_depth_payload["error"] == "invalid_depth"
    assert invalid_depth_payload["inbound_edges"] == []


def test_ego_tool_returns_cli_helper_payload_shape(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _, b_id, _ = _populate_backlinks_graph(store)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "ego")
    assert tool.inputSchema["properties"]["depth"]["maximum"] == 3

    response = asyncio.run(mcp_server.call_tool("ego", {"unit_id": b_id, "depth": 9}))
    payload = json.loads(response[0].text)

    assert set(payload) == {"unit_id", "center", "metrics", "relation_counts", "depth"}
    assert payload["center"]["title"] == "Node B"
    assert payload["depth"] == 3
    assert payload["metrics"]["in_degree"] == 1
    assert payload["metrics"]["out_degree"] == 1
    assert payload["relation_counts"] == {"builds_on": 1, "inspires": 1}

    missing = asyncio.run(mcp_server.call_tool("ego", {"unit_id": "missing"}))
    missing_payload = json.loads(missing[0].text)
    assert missing_payload["error"] == "unit_not_found"
    assert missing_payload["metrics"] == {}
    assert missing_payload["relation_counts"] == {}


def test_shortest_path_tool_returns_structured_path_and_errors(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    a_id, b_id, c_id = _populate_backlinks_graph(store)
    isolated = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="isolated",
            source_entity_type="knowledge_item",
            title="Isolated",
            content="Outside the path",
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "shortest_path")
    assert set(tool.inputSchema["required"]) == {"from_unit_id", "to_unit_id"}
    assert "from_unit_id" in tool.inputSchema["properties"]
    assert "to_unit_id" in tool.inputSchema["properties"]

    response = asyncio.run(
        mcp_server.call_tool(
            "shortest_path",
            {"from_unit_id": a_id, "to_unit_id": c_id},
        )
    )
    payload = json.loads(response[0].text)
    assert [unit["id"] for unit in payload["path"]] == [a_id, b_id, c_id]
    assert [unit["title"] for unit in payload["path"]] == [
        "Node A",
        "Node B",
        "Node C",
    ]
    assert payload["edges"][0]["relation"] == "builds_on"
    assert payload["edges"][0]["weight"] == 0.75
    assert payload["edges"][0]["source"] == "manual"
    assert payload["edges"][1]["relation"] == "inspires"
    assert payload["edges"][1]["source"] == "inferred"

    missing = asyncio.run(
        mcp_server.call_tool(
            "shortest_path",
            {"from_unit_id": "missing", "to_unit_id": c_id},
        )
    )
    missing_payload = json.loads(missing[0].text)
    assert missing_payload["error"] == "unit_not_found"
    assert missing_payload["missing_unit_ids"] == ["missing"]
    assert missing_payload["path"] == []
    assert missing_payload["edges"] == []

    disconnected = asyncio.run(
        mcp_server.call_tool(
            "shortest_path",
            {"from_unit_id": a_id, "to_unit_id": isolated.id},
        )
    )
    disconnected_payload = json.loads(disconnected[0].text)
    assert disconnected_payload["error"] == "not_connected"
    assert disconnected_payload["path"] == []
    assert disconnected_payload["edges"] == []


def test_export_jsonl_tool_returns_counts_and_writes_selected_records(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    export_path = tmp_path / "graph.jsonl"
    store = Store(str(db_path))
    first = store.insert_unit(
        KnowledgeUnit(
            id="unit-a",
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title="Node A",
            content="First node",
            created_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        )
    )
    second = store.insert_unit(
        KnowledgeUnit(
            id="unit-b",
            source_project=SourceProject.MAX,
            source_id="b",
            source_entity_type="insight",
            title="Node B",
            content="Second node",
            created_at=datetime.fromisoformat("2026-01-02T00:00:00+00:00"),
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            id="edge-a",
            from_unit_id=first.id,
            to_unit_id=second.id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.MANUAL,
            created_at=datetime.fromisoformat("2026-01-03T00:00:00+00:00"),
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "export_jsonl" for tool in tools)

    response = asyncio.run(
        mcp_server.call_tool(
            "export_jsonl",
            {"path": str(export_path), "record_type": "units"},
        )
    )
    payload = json.loads(response[0].text)
    records = [json.loads(line) for line in export_path.read_text().splitlines()]

    assert payload["path"] == str(export_path)
    assert payload["record_type"] == "units"
    assert payload["records_exported"] == 2
    assert payload["units_exported"] == 2
    assert payload["edges_exported"] == 0
    assert [record["record_type"] for record in records] == ["unit", "unit"]
    assert [record["id"] for record in records] == ["unit-a", "unit-b"]


def test_export_context_pack_tool_is_advertised_with_unit_id_schema():
    tools = asyncio.run(mcp_server.list_tools())
    export_tool = next(tool for tool in tools if tool.name == "export_context_pack")

    assert "required" not in export_tool.inputSchema
    assert export_tool.inputSchema["properties"]["unit_ids"]["type"] == "array"
    assert export_tool.inputSchema["properties"]["unit_ids"]["items"] == {"type": "string"}
    assert export_tool.inputSchema["properties"]["query"]["type"] == "string"
    assert export_tool.inputSchema["properties"]["tag"]["type"] == "string"
    assert export_tool.inputSchema["properties"]["limit"]["minimum"] == 1
    assert export_tool.inputSchema["properties"]["max_chars"]["minimum"] == 1
    assert export_tool.inputSchema["properties"]["options"]["type"] == "object"


def test_export_obsidian_tool_exports_same_vault_structure_as_cli(tmp_path, monkeypatch):
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


def test_import_obsidian_tool_returns_counts_and_imports_wikilinks(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    vault = tmp_path / "vault"
    nested = vault / "Notes" / "Nested"
    nested.mkdir(parents=True)
    (vault / "Notes" / "Alpha.md").write_text(
        "---\n"
        "title: Alpha MCP\n"
        "tags: [front]\n"
        "---\n"
        "Alpha links to [[Beta]] with #inline.\n",
        encoding="utf-8",
    )
    (nested / "Beta.md").write_text("Beta body.\n", encoding="utf-8")

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    import_tool = next(tool for tool in tools if tool.name == "import_obsidian")
    assert set(import_tool.inputSchema["required"]) == {"vault_path"}
    assert import_tool.inputSchema["properties"]["include_tags"]["default"] is True

    response = asyncio.run(
        mcp_server.call_tool(
            "import_obsidian",
            {
                "vault_path": str(vault),
                "folder": "Notes",
                "source_project": "obsidian",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["notes_found"] == 2
    assert payload["wikilink_edges_found"] == 1
    assert payload["units_inserted"] == 2
    assert payload["edges_inserted"] == 1

    store = Store(str(db_path))
    try:
        alpha = store.get_unit_by_source("obsidian", "Notes/Alpha.md", "markdown_note")
        beta = store.get_unit_by_source("obsidian", "Notes/Nested/Beta.md", "markdown_note")
        assert alpha is not None
        assert beta is not None
        assert alpha.title == "Alpha MCP"
        assert alpha.metadata["path"] == "Notes/Alpha.md"
        assert alpha.tags == ["front", "inline"]
        edges = store.get_all_edges()
        assert len(edges) == 1
        assert edges[0].from_unit_id == alpha.id
        assert edges[0].to_unit_id == beta.id
    finally:
        store.close()


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
    export_response = asyncio.run(mcp_server.call_tool("export_json", {"path": str(backup_path)}))
    export_payload = json.loads(export_response[0].text)

    assert export_payload["units_exported"] == 2
    assert export_payload["edges_exported"] == 1
    assert backup_path.exists()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(target_db)))
    import_response = asyncio.run(mcp_server.call_tool("import_json", {"path": str(backup_path)}))
    import_payload = json.loads(import_response[0].text)

    assert import_payload["units_inserted"] == 2
    assert import_payload["edges_inserted"] == 1

    second_response = asyncio.run(mcp_server.call_tool("import_json", {"path": str(backup_path)}))
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
    response = asyncio.run(mcp_server.call_tool("export_graphml", {"path": str(graphml_path)}))
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


def test_export_turtle_tool_returns_path_counts_and_base_uri(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    turtle_path = tmp_path / "graph.ttl"

    store = Store(str(db_path))
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="a",
            source_entity_type="knowledge_node",
            title='Node "A"',
            content="First searchable node",
            tags=["energy", "solar"],
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
    assert any(tool.name == "export_turtle" for tool in tools)

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    response = asyncio.run(
        mcp_server.call_tool(
            "export_turtle",
            {
                "path": str(turtle_path),
                "base_uri": "https://example.test/unit/",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "path": str(turtle_path),
        "node_count": 2,
        "edge_count": 1,
        "base_uri": "https://example.test/unit/",
    }
    text = turtle_path.read_text()
    assert "@prefix graph: <https://graph.local/schema#> ." in text
    assert f"<https://example.test/unit/{a.id}>" in text
    assert 'graph:title "Node \\"A\\""' in text
    assert 'graph:tag "energy"' in text
    assert f"graph:builds_on <https://example.test/unit/{b.id}>" in text


def test_export_markdown_tool_returns_counts_path_and_filters(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    export_dir = tmp_path / "markdown"

    store = Store(str(db_path))
    matching = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="matching",
            source_entity_type="insight",
            title="Matching Unit",
            content="Matching insight",
            content_type=ContentType.INSIGHT,
            tags=["energy", "solar"],
            metadata={"source": "test"},
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="other",
            source_entity_type="knowledge_node",
            title="Other Unit",
            content="Other content",
            content_type=ContentType.INSIGHT,
            tags=["energy"],
        )
    )
    store.close()

    export_dir.mkdir()
    (export_dir / "stale.md").write_text("old")

    tools = asyncio.run(mcp_server.list_tools())
    export_tool = next(tool for tool in tools if tool.name == "export_markdown")
    assert export_tool.inputSchema["properties"]["clean"]["default"] is False

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    response = asyncio.run(
        mcp_server.call_tool(
            "export_markdown",
            {
                "path": str(export_dir),
                "clean": True,
                "tag": "solar",
                "source_project": "max",
                "content_type": "insight",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["path"] == str(export_dir)
    assert payload["units_exported"] == 1
    assert payload["files_written"] == 1
    assert payload["filters"] == {
        "tag": "solar",
        "source_project": "max",
        "content_type": "insight",
    }
    assert not (export_dir / "stale.md").exists()

    files = list(export_dir.glob("*.md"))
    assert len(files) == 1
    raw_front_matter = files[0].read_text().split("---", 2)[1]
    front_matter = yaml.safe_load(raw_front_matter)
    assert front_matter["id"] == matching.id
    assert front_matter["source_project"] == "max"
    assert front_matter["source_id"] == "matching"
    assert front_matter["source_entity_type"] == "insight"
    assert front_matter["content_type"] == "insight"
    assert front_matter["tags"] == ["energy", "solar"]
    assert front_matter["metadata"] == {"source": "test"}


def test_export_mermaid_tool_returns_path_counts_and_capped_flag(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    mermaid_path = tmp_path / "graph.md"

    store = Store(str(db_path))
    a, b, c = _populate_backlinks_graph(store)
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="isolated",
            source_entity_type="knowledge_item",
            title="Isolated",
            content="Outside the export cap",
        )
    )
    store.close()

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "export_mermaid" for tool in tools)

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    response = asyncio.run(
        mcp_server.call_tool(
            "export_mermaid",
            {
                "path": str(mermaid_path),
                "unit_id": a,
                "depth": 2,
                "limit": 2,
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "path": str(mermaid_path),
        "node_count": 2,
        "edge_count": 1,
        "capped": True,
        "depth": 2,
        "center_unit_id": a,
    }
    text = mermaid_path.read_text()
    assert text.startswith("```mermaid\ngraph TD\n")
    assert "builds_on" in text
    assert c not in text


def test_export_cytoscape_tool_returns_path_counts_and_writes_elements(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    cytoscape_path = tmp_path / "graph.cy.json"

    store = Store(str(db_path))
    a, b, c = _populate_backlinks_graph(store)
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="isolated",
            source_entity_type="knowledge_item",
            title="Isolated",
            content="Outside the export cap",
        )
    )
    store.close()

    tools = asyncio.run(mcp_server.list_tools())
    export_tool = next(tool for tool in tools if tool.name == "export_cytoscape")
    assert export_tool.inputSchema["properties"]["limit"]["default"] == 100
    assert export_tool.inputSchema["properties"]["depth"]["maximum"] == 3

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    response = asyncio.run(
        mcp_server.call_tool(
            "export_cytoscape",
            {
                "path": str(cytoscape_path),
                "unit_id": a,
                "depth": 9,
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "path": str(cytoscape_path),
        "node_count": 3,
        "edge_count": 2,
        "mode": "neighborhood",
        "capped": False,
        "depth": 3,
        "center_unit_id": a,
    }
    exported = json.loads(cytoscape_path.read_text())
    assert {node["data"]["id"] for node in exported["elements"]["nodes"]} == {a, b, c}
    edge_data = {edge["data"]["relation"]: edge["data"] for edge in exported["elements"]["edges"]}
    assert edge_data["builds_on"]["id"]
    assert edge_data["builds_on"]["source"] == a
    assert edge_data["builds_on"]["target"] == b
    assert edge_data["builds_on"]["edge_source"] == "manual"


def test_export_neighborhood_tool_returns_path_counts_and_writes_json(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    export_path = tmp_path / "neighborhood.json"

    store = Store(str(db_path))
    a, b, c = _populate_backlinks_graph(store)
    isolated = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="isolated",
            source_entity_type="knowledge_item",
            title="Isolated",
            content="Outside the neighborhood",
        )
    )
    store.close()

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "export_neighborhood" for tool in tools)

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    response = asyncio.run(
        mcp_server.call_tool(
            "export_neighborhood",
            {"unit_id": a, "path": str(export_path), "depth": 2},
        )
    )
    payload = json.loads(response[0].text)

    assert payload == {
        "path": str(export_path),
        "unit_count": 3,
        "edge_count": 2,
        "depth": 2,
        "center_unit_id": a,
    }
    exported = json.loads(export_path.read_text())
    assert exported["center"]["id"] == a
    assert {unit["id"] for unit in exported["units"]} == {a, b, c}
    assert isolated.id not in {unit["id"] for unit in exported["units"]}
    assert {edge["relation"] for edge in exported["edges"]} == {
        "builds_on",
        "inspires",
    }

    missing_response = asyncio.run(
        mcp_server.call_tool(
            "export_neighborhood",
            {"unit_id": "missing", "path": str(tmp_path / "missing.json")},
        )
    )
    missing_payload = json.loads(missing_response[0].text)
    assert missing_payload["error"] == "unit_not_found"
    assert missing_payload["unit_id"] == "missing"


def test_export_report_tool_writes_markdown_with_same_sections(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    report_path = tmp_path / "graph-report.md"

    store = Store(str(db_path))
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="report-a",
            source_entity_type="knowledge_node",
            title="Report Node A",
            content="First report node",
            tags=["energy", "solar"],
            utility_score=0.9,
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="report-b",
            source_entity_type="insight",
            title="Report Node B",
            content="Second report node",
            content_type=ContentType.INSIGHT,
            tags=["energy", "storage"],
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

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "export_report" for tool in tools)

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))
    response = asyncio.run(
        mcp_server.call_tool(
            "export_report",
            {"path": str(report_path), "limit": 2},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["path"] == str(report_path)
    assert payload["section_counts"]["central"] == 2
    assert payload["section_counts"]["tags"] == 2
    assert payload["section_counts"]["cross_project"] == 1

    report = report_path.read_text()
    for heading in [
        "## Stats",
        "## Top Central Nodes",
        "## Bridge Nodes",
        "## Largest Clusters",
        "## Gap Candidates",
        "## Top Tags",
        "## Cross-Project Connections",
    ]:
        assert heading in report
    assert "Report Node A" in report
    assert "Report Node B" in report
    assert "energy: 2 units" in report
    assert "forty_two <-> max: 1 edges" in report


def test_analyze_k_core_tool_catalog_and_payload(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    a = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="core-a",
            source_entity_type="insight",
            title="Core A",
            content="Core A content",
        )
    )
    b = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="core-b",
            source_entity_type="insight",
            title="Core B",
            content="Core B content",
        )
    )
    c = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="core-c",
            source_entity_type="knowledge_node",
            title="Core C",
            content="Core C content",
        )
    )
    d = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="leaf-d",
            source_entity_type="knowledge_item",
            title="Leaf D",
            content="Leaf content",
        )
    )
    for from_unit, to_unit in [(a, b), (b, c), (c, a), (c, d)]:
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=from_unit.id,
                to_unit_id=to_unit.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
    expected = GraphService(store).analyze_k_core(k=2, limit=10)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    analyze_tool = next(tool for tool in tools if tool.name == "analyze_k_core")
    schema = analyze_tool.inputSchema["properties"]
    assert schema["k"]["default"] is None
    assert schema["k"]["minimum"] == 0
    assert schema["limit"]["default"] == 50
    assert schema["include_units"]["default"] is True

    response = asyncio.run(
        mcp_server.call_tool("analyze_k_core", {"k": 2, "limit": 10})
    )
    payload = json.loads(response[0].text)

    assert payload == expected
    assert payload["k"] == 2
    assert payload["node_count"] == 3
    assert payload["edge_count"] == 3
    assert {node["id"] for node in payload["nodes"]} == {a.id, b.id, c.id}
    assert d.id not in {node["id"] for node in payload["nodes"]}
    assert all(node["core_number"] >= 2 for node in payload["nodes"])
    assert all("unit" in node for node in payload["nodes"])


def test_analyze_k_core_tool_respects_include_units_false(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    units = [
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=f"core-{index}",
                source_entity_type="insight",
                title=f"Core {index}",
                content=f"Core content {index}",
            )
        )
        for index in range(3)
    ]
    for left, right in [(0, 1), (1, 2), (2, 0)]:
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=units[left].id,
                to_unit_id=units[right].id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool(
            "analyze_k_core",
            {"k": 2, "limit": 10, "include_units": False},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["filters"]["include_units"] is False
    assert payload["nodes"]
    assert all("unit" not in node for node in payload["nodes"])


def test_analyze_k_core_tool_serializes_validation_errors(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    Store(str(db_path)).close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    invalid_k = asyncio.run(mcp_server.call_tool("analyze_k_core", {"k": -1}))
    invalid_k_payload = json.loads(invalid_k[0].text)
    assert invalid_k_payload == {
        "error": "invalid_k",
        "message": "k must be a non-negative integer.",
        "arguments": {"k": -1, "limit": 50, "include_units": True},
    }

    invalid_limit = asyncio.run(
        mcp_server.call_tool("analyze_k_core", {"k": 2, "limit": "many"})
    )
    invalid_limit_payload = json.loads(invalid_limit[0].text)
    assert invalid_limit_payload == {
        "error": "invalid_limit",
        "message": "limit must be a positive integer.",
        "arguments": {"k": 2, "limit": "many", "include_units": True},
    }


def test_analyze_relation_motifs_tool_is_available_from_mcp_server(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "graph.db"
    Store(str(db_path)).close()
    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "analyze_relation_motifs")
    assert tool.inputSchema["properties"]["limit"]["default"] == 20
    assert tool.inputSchema["properties"]["relation_sequence"]["items"]["enum"]

    response = asyncio.run(mcp_server.call_tool("analyze_relation_motifs", {}))
    payload = json.loads(response[0].text)

    assert payload["motifs"] == []
    assert payload["stats"]["total_paths"] == 0


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

    summary_response = asyncio.run(mcp_server.call_tool("analyze_tags", {"limit": 2}))
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


def test_tag_graph_tool_exposes_filters_and_returns_json_payload(tmp_path, monkeypatch):
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
    tag_graph_tool = next(tool for tool in tools if tool.name == "tag_graph")
    assert "source_project" in tag_graph_tool.inputSchema["properties"]
    assert "content_type" in tag_graph_tool.inputSchema["properties"]
    assert "min_count" in tag_graph_tool.inputSchema["properties"]

    response = asyncio.run(
        mcp_server.call_tool(
            "tag_graph",
            {
                "source_project": "max",
                "content_type": "insight",
                "min_count": 2,
                "limit": 5,
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["nodes"] == [
        {"id": "energy", "tag": "energy", "unit_count": 2},
        {"id": "storage", "tag": "storage", "unit_count": 2},
    ]
    assert [
        (edge["source"], edge["target"], edge["co_occurrence_count"])
        for edge in payload["edges"]
    ] == [("energy", "storage", 2)]
    assert payload["edges"][0]["representative_unit_ids"]
    assert payload["filters"] == {
        "source_project": "max",
        "content_type": "insight",
        "min_count": 2,
        "limit": 5,
    }


def test_timeline_tool_matches_service_schema_and_validates_inputs(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_timeline_graph(store)
    store.close()

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )

    tools = asyncio.run(mcp_server.list_tools())
    timeline_tool = next(tool for tool in tools if tool.name == "timeline")
    assert timeline_tool.inputSchema["properties"]["bucket"]["enum"] == [
        "day",
        "week",
        "month",
        "year",
    ]
    assert timeline_tool.inputSchema["properties"]["field"]["enum"] == [
        "created_at",
        "ingested_at",
        "updated_at",
    ]

    response = asyncio.run(
        mcp_server.call_tool(
            "timeline",
            {
                "bucket": "month",
                "source_project": "max",
                "content_type": "insight",
            },
        )
    )
    payload = json.loads(response[0].text)
    assert payload["field"] == "created_at"
    assert payload["total"] == 2
    assert [item["bucket"] for item in payload["buckets"]] == ["2026-01", "2026-02"]
    assert payload["buckets"][1]["content_types"] == {"insight": 1}

    invalid_bucket = asyncio.run(mcp_server.call_tool("timeline", {"bucket": "quarter"}))
    invalid_bucket_payload = json.loads(invalid_bucket[0].text)
    assert invalid_bucket_payload["error"] == "invalid_bucket"
    assert "Unsupported timeline bucket" in invalid_bucket_payload["message"]

    invalid_field = asyncio.run(mcp_server.call_tool("timeline", {"field": "deleted_at"}))
    invalid_field_payload = json.loads(invalid_field[0].text)
    assert invalid_field_payload["error"] == "invalid_field"
    assert "Unsupported timeline field" in invalid_field_payload["message"]


def test_freshness_summary_tool_matches_service_schema(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_timeline_graph(store)
    store.close()

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )

    tools = asyncio.run(mcp_server.list_tools())
    freshness_tool = next(tool for tool in tools if tool.name == "freshness_summary")
    assert freshness_tool.inputSchema["properties"]["field"]["enum"] == [
        "created_at",
        "ingested_at",
        "updated_at",
    ]

    response = asyncio.run(
        mcp_server.call_tool(
            "freshness_summary",
            {
                "field": "created_at",
                "as_of": "2026-05-01T00:00:00+00:00",
            },
        )
    )
    payload = json.loads(response[0].text)
    expected_store = Store(str(db_path))
    expected = GraphService(expected_store).freshness_summary(
        field="created_at",
        as_of="2026-05-01T00:00:00+00:00",
    )
    expected_store.close()
    assert payload == expected


def test_suggest_tag_synonyms_tool_matches_service_structure(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_tag_synonym_graph(store)
    expected = GraphService(store).suggest_tag_synonyms(limit=5, min_similarity=0.8)
    store.close()

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )

    tools = asyncio.run(mcp_server.list_tools())
    synonym_tool = next(tool for tool in tools if tool.name == "suggest_tag_synonyms")
    assert synonym_tool.inputSchema["properties"]["limit"]["default"] == 20
    assert synonym_tool.inputSchema["properties"]["min_similarity"]["default"] == 0.8

    response = asyncio.run(
        mcp_server.call_tool(
            "suggest_tag_synonyms",
            {"limit": 5, "min_similarity": 0.8},
        )
    )
    payload = json.loads(response[0].text)

    assert payload == expected
    assert payload["suggestions"][0]["canonical_candidate"] == "ai-agent"
    assert {variant["tag"] for variant in payload["suggestions"][0]["variants"]} == {
        "ai-agent",
        "ai_agent",
        "AI Agents",
    }


def test_suggest_tags_tool_returns_suggestions_without_mutating(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    target = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="target-suggest-tags",
            source_entity_type="insight",
            title="Solar battery planning",
            content="Battery storage roadmap for grid operations.",
            content_type=ContentType.INSIGHT,
            tags=["solar"],
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="battery-tag-source",
            source_entity_type="insight",
            title="Battery source",
            content="Existing battery source",
            tags=["battery"],
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="storage-tag-source",
            source_entity_type="knowledge_node",
            title="Storage source",
            content="Existing storage source",
            tags=["storage"],
        )
    )
    store.close()

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )

    tools = asyncio.run(mcp_server.list_tools())
    suggest_tool = next(tool for tool in tools if tool.name == "suggest_tags")
    assert "unit_id" in suggest_tool.inputSchema["required"]

    response = asyncio.run(
        mcp_server.call_tool(
            "suggest_tags",
            {"unit_id": target.id, "limit": 5, "min_score": 0.25},
        )
    )
    payload = json.loads(response[0].text)

    assert [item["tag"] for item in payload["suggestions"]] == ["battery", "storage"]
    reopened = Store(str(db_path))
    assert reopened.get_unit(target.id).tags == ["solar"]  # type: ignore[union-attr]
    reopened.close()


def test_rename_tag_tool_dry_run_and_execute_match_service_counts(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_tag_synonym_graph(store)
    store.close()

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )

    tools = asyncio.run(mcp_server.list_tools())
    rename_tool = next(tool for tool in tools if tool.name == "rename_tag")
    assert rename_tool.inputSchema["properties"]["dry_run"]["default"] is False
    assert "source_project" in rename_tool.inputSchema["properties"]
    assert "content_type" in rename_tool.inputSchema["properties"]

    dry_response = asyncio.run(
        mcp_server.call_tool(
            "rename_tag",
            {
                "old_tag": "ai_agent",
                "new_tag": "ai-agent",
                "source_project": "max",
                "content_type": "insight",
                "dry_run": True,
            },
        )
    )
    dry_payload = json.loads(dry_response[0].text)
    assert dry_payload["dry_run"] is True
    assert dry_payload["matched_count"] == 1
    assert dry_payload["updated_count"] == 1
    assert dry_payload["changed_count"] == 1
    assert dry_payload["affected_unit_ids"]
    assert dry_payload["sample_units"][0]["title"] == "Agent underscore"

    store = Store(str(db_path))
    unit = store.get_unit_by_source("max", "agent-underscore", "insight")
    assert unit is not None
    assert unit.tags == ["ai_agent"]
    store.close()

    execute_response = asyncio.run(
        mcp_server.call_tool(
            "rename_tag",
            {
                "old_tag": "ai_agent",
                "new_tag": "ai-agent",
                "source_project": "max",
                "content_type": "insight",
            },
        )
    )
    execute_payload = json.loads(execute_response[0].text)
    assert execute_payload["dry_run"] is False
    assert execute_payload["matched_count"] == dry_payload["matched_count"]
    assert execute_payload["updated_count"] == dry_payload["updated_count"]
    assert execute_payload["changed_count"] == dry_payload["changed_count"]
    assert execute_payload["affected_unit_ids"] == dry_payload["affected_unit_ids"]
    assert execute_payload["sample_units"] == dry_payload["sample_units"]

    store = Store(str(db_path))
    unit = store.get_unit_by_source("max", "agent-underscore", "insight")
    assert unit is not None
    assert unit.tags == ["ai-agent"]
    store.close()


def test_remove_tag_tool_dry_run_and_execute_reindexes_fts(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    target = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="mcp-remove-target",
            source_entity_type="insight",
            title="MCP remove target",
            content="MCP removable content",
            content_type=ContentType.INSIGHT,
            tags=["retire_tag", "keep"],
        )
    )
    other = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="mcp-remove-other",
            source_entity_type="knowledge_node",
            title="MCP remove other",
            content="Other removable content",
            content_type=ContentType.FINDING,
            tags=["retire_tag"],
        )
    )
    for unit in [target, other]:
        store.fts_index_unit(unit)
    store.close()

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )

    tools = asyncio.run(mcp_server.list_tools())
    remove_tool = next(tool for tool in tools if tool.name == "remove_tag")
    assert remove_tool.inputSchema["properties"]["dry_run"]["default"] is False
    assert "source_project" in remove_tool.inputSchema["properties"]
    assert "limit" in remove_tool.inputSchema["properties"]

    dry_response = asyncio.run(
        mcp_server.call_tool(
            "remove_tag",
            {
                "tag": "retire_tag",
                "source_project": "max",
                "content_type": "insight",
                "limit": 10,
                "dry_run": True,
            },
        )
    )
    dry_payload = json.loads(dry_response[0].text)
    assert dry_payload["dry_run"] is True
    assert dry_payload["matched_count"] == 1
    assert dry_payload["removed_count"] == 1
    assert dry_payload["changed_units"][0]["id"] == target.id

    store = Store(str(db_path))
    unit = store.get_unit(target.id)
    assert unit is not None
    assert unit.tags == ["retire_tag", "keep"]
    assert target.id in {row["unit_id"] for row in store.fts_search("retire_tag")}
    store.close()

    execute_response = asyncio.run(
        mcp_server.call_tool(
            "remove_tag",
            {
                "tag": "retire_tag",
                "source_project": "max",
                "content_type": "insight",
                "limit": 10,
            },
        )
    )
    execute_payload = json.loads(execute_response[0].text)
    assert execute_payload["dry_run"] is False
    assert execute_payload["matched_count"] == dry_payload["matched_count"]
    assert execute_payload["removed_count"] == dry_payload["removed_count"]
    assert execute_payload["affected_unit_ids"] == dry_payload["affected_unit_ids"]

    store = Store(str(db_path))
    unit = store.get_unit(target.id)
    skipped = store.get_unit(other.id)
    assert unit is not None
    assert skipped is not None
    assert unit.tags == ["keep"]
    assert skipped.tags == ["retire_tag"]
    assert target.id not in {row["unit_id"] for row in store.fts_search("retire_tag")}
    store.close()


def test_apply_tags_to_search_tool_returns_counts_summaries_and_updates(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_tags_graph(store)
    for unit in store.get_all_units(limit=100):
        store.fts_index_unit(unit)
    store.close()

    monkeypatch.setattr(
        mcp_server,
        "_get_store",
        lambda: Store(str(db_path)),
    )

    tools = asyncio.run(mcp_server.list_tools())
    apply_tool = next(tool for tool in tools if tool.name == "apply_tags_to_search")
    assert apply_tool.inputSchema["properties"]["dry_run"]["default"] is False
    assert apply_tool.inputSchema["properties"]["mode"]["default"] == "fulltext"
    assert "source_project" in apply_tool.inputSchema["properties"]

    dry_response = asyncio.run(
        mcp_server.call_tool(
            "apply_tags_to_search",
            {
                "query": "Solar",
                "add": ["curated", "curated"],
                "remove": ["energy"],
                "source_project": "max",
                "tag": "energy",
                "dry_run": True,
            },
        )
    )
    dry_payload = json.loads(dry_response[0].text)
    assert dry_payload["dry_run"] is True
    assert dry_payload["matched_count"] == 1
    assert dry_payload["affected_count"] == 1
    assert dry_payload["add_tags"] == ["curated"]
    assert dry_payload["affected_units"][0]["old_tags"] == [
        "energy",
        "solar",
        "storage",
    ]
    assert dry_payload["affected_units"][0]["new_tags"] == [
        "solar",
        "storage",
        "curated",
    ]

    store = Store(str(db_path))
    unit = store.get_unit_by_source("max", "solar-storage", "insight")
    assert unit is not None
    assert unit.tags == ["energy", "solar", "storage"]
    store.close()

    execute_response = asyncio.run(
        mcp_server.call_tool(
            "apply_tags_to_search",
            {
                "query": "Solar",
                "add": ["curated"],
                "remove": ["energy"],
                "source_project": "max",
                "tag": "energy",
            },
        )
    )
    execute_payload = json.loads(execute_response[0].text)
    assert execute_payload["dry_run"] is False
    assert execute_payload["affected_count"] == dry_payload["affected_count"]

    store = Store(str(db_path))
    unit = store.get_unit_by_source("max", "solar-storage", "insight")
    other = store.get_unit_by_source("forty_two", "solar-grid", "knowledge_node")
    assert unit is not None
    assert other is not None
    assert unit.tags == ["solar", "storage", "curated"]
    assert other.tags == ["energy", "solar", "grid"]
    assert store.fts_search("curated")[0]["unit_id"] == unit.id
    store.close()


def test_analyze_duplicates_tool_returns_reasons_units_and_filters(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_duplicates_graph(store)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    analyze_tool = next(tool for tool in tools if tool.name == "analyze_duplicates")
    assert "source_project" in analyze_tool.inputSchema["properties"]
    assert "content_type" in analyze_tool.inputSchema["properties"]
    assert analyze_tool.inputSchema["properties"]["min_title_similarity"]["default"] == 0.92

    response = asyncio.run(
        mcp_server.call_tool(
            "analyze_duplicates",
            {
                "source_project": "max",
                "content_type": "insight",
                "min_title_similarity": 0.85,
            },
        )
    )
    payload = json.loads(response[0].text)
    by_reason = {item["reason"]: item for item in payload["groups"]}
    assert set(by_reason) == {"canonical_url", "title_similarity"}
    assert all(item["id"].startswith("dup_") for item in payload["groups"])
    assert "canonical_url" in by_reason["canonical_url"]["reasons"]
    assert {unit["source_id"] for unit in by_reason["canonical_url"]["units"]} == {
        "canonical-a",
        "canonical-b",
    }
    assert by_reason["title_similarity"]["score"] >= 0.85
    assert {unit["source_id"] for unit in by_reason["title_similarity"]["units"]} == {
        "title-a",
        "title-b",
    }
    assert all(
        unit["source_project"] == "max" for item in payload["groups"] for unit in item["units"]
    )


def test_review_queue_tool_returns_ranked_items_reasons_and_filters(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    old_id, recent_id = _populate_review_queue_graph(store)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    review_tool = next(tool for tool in tools if tool.name == "review_queue")
    assert "source_project" in review_tool.inputSchema["properties"]
    assert "content_type" in review_tool.inputSchema["properties"]

    response = asyncio.run(
        mcp_server.call_tool(
            "review_queue",
            {
                "source_project": "max",
                "content_type": "insight",
            },
        )
    )
    payload = json.loads(response[0].text)
    assert payload["filters"] == {
        "source_project": "max",
        "content_type": "insight",
    }
    assert [item["unit"]["id"] for item in payload["queue"]] == [
        old_id,
        recent_id,
    ]
    assert payload["queue"][0]["score"] > payload["queue"][1]["score"]
    assert {"age", "isolated", "utility_score", "unreviewed"} <= {
        reason["code"] for reason in payload["queue"][0]["reasons"]
    }


def test_suggest_edges_tool_returns_parseable_candidate_json(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    from_id, to_id = _populate_edge_suggestion_graph(store)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    suggest_tool = next(tool for tool in tools if tool.name == "suggest_edges")
    assert suggest_tool.inputSchema["properties"]["limit"]["default"] == 20
    assert suggest_tool.inputSchema["properties"]["min_score"]["default"] == 0.4
    assert "source_project" in suggest_tool.inputSchema["properties"]

    response = asyncio.run(
        mcp_server.call_tool(
            "suggest_edges",
            {
                "source_project": "max",
                "min_score": 0.4,
                "limit": 5,
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["filters"] == {"source_project": "max"}
    assert len(payload["candidates"]) == 1
    candidate = payload["candidates"][0]
    assert candidate["from_id"] == from_id
    assert candidate["to_id"] == to_id
    assert candidate["score"] >= 0.8
    assert any(reason.startswith("shared tags:") for reason in candidate["reasons"])
    assert any(reason.startswith("shared links:") for reason in candidate["reasons"])
    assert {"from_id", "to_id", "score", "reasons"} <= set(candidate)


def test_analyze_links_tool_returns_same_structured_payload(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    _populate_links_graph(store)
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    analyze_tool = next(tool for tool in tools if tool.name == "analyze_links")
    assert "domain" in analyze_tool.inputSchema["properties"]
    assert "limit" in analyze_tool.inputSchema["properties"]

    response = asyncio.run(
        mcp_server.call_tool(
            "analyze_links",
            {"domain": "example.com", "limit": 5},
        )
    )
    payload = json.loads(response[0].text)

    assert payload["filters"] == {"domain": "example.com"}
    assert payload["domains"][0]["domain"] == "example.com"
    assert payload["domains"][0]["count"] == 2
    assert payload["links"][0]["url"] == "https://example.com/docs"
    assert payload["links"][0]["count"] == 2
    assert {occurrence["source_project"] for occurrence in payload["links"][0]["occurrences"]} == {
        "max",
        "forty_two",
    }


def test_analyze_source_coverage_tool_returns_service_payload(tmp_path, monkeypatch):
    from graph.graph.service import GraphService

    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
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
            source_project=SourceProject.MAX,
            source_id="b",
            source_entity_type="insight",
            title="Node B",
            content="Second node",
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="c",
            source_entity_type="knowledge_item",
            title="Node C",
            content="Isolated node",
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=a.id,
            to_unit_id=b.id,
            relation=EdgeRelation.BUILDS_ON,
        )
    )
    store.upsert_sync_state(
        SyncState(
            source_project="sota",
            source_entity_type="paper",
            last_sync_at="2026-04-24T01:00:00+00:00",
            last_source_id="paper-1",
            items_synced=5,
        )
    )
    expected = GraphService(store).analyze_source_coverage()
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    assert any(tool.name == "analyze_source_coverage" for tool in tools)

    response = asyncio.run(mcp_server.call_tool("analyze_source_coverage", {}))
    payload = json.loads(response[0].text)
    assert payload == expected
    by_source = {
        (item["source_project"], item["source_entity_type"]): item for item in payload["sources"]
    }
    assert by_source[("presence", "knowledge_item")]["orphan_count"] == 1
    assert by_source[("sota", "paper")]["unit_count"] == 0


def test_find_orphan_units_tool_returns_counts_filters_and_units(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    source = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="source",
            source_entity_type="knowledge_node",
            title="Source node",
            content="Has outgoing edge",
            content_type=ContentType.FINDING,
            tags=["energy"],
        )
    )
    target = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="target",
            source_entity_type="knowledge_node",
            title="Target node",
            content="Has incoming edge",
            content_type=ContentType.FINDING,
            tags=["energy"],
        )
    )
    max_orphan = store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="max-orphan",
            source_entity_type="insight",
            title="Max orphan",
            content="No edges",
            content_type=ContentType.INSIGHT,
            tags=["energy"],
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="presence-orphan",
            source_entity_type="knowledge_item",
            title="Presence orphan",
            content="No edges",
            content_type=ContentType.ARTIFACT,
            tags=["archive"],
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=source.id,
            to_unit_id=target.id,
            relation=EdgeRelation.BUILDS_ON,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    orphan_tool = next(tool for tool in tools if tool.name == "find_orphan_units")
    assert orphan_tool.inputSchema["properties"]["limit"]["default"] == 20
    assert "source_project" in orphan_tool.inputSchema["properties"]

    response = asyncio.run(
        mcp_server.call_tool(
            "find_orphan_units",
            {
                "source_project": "max",
                "content_type": "insight",
                "tag": "energy",
                "limit": 5,
            },
        )
    )
    payload = json.loads(response[0].text)

    assert payload["total_count"] == 1
    assert payload["returned_count"] == 1
    assert payload["filters"] == {
        "source_project": "max",
        "content_type": "insight",
        "tag": "energy",
        "limit": 5,
    }
    assert [unit["id"] for unit in payload["units"]] == [max_orphan.id]


def test_weak_component_summary_tool_is_registered():
    tools = asyncio.run(mcp_server.list_tools())
    tool = next(tool for tool in tools if tool.name == "weak_component_summary")

    assert tool.inputSchema["properties"]["limit"]["default"] == 20
    assert tool.inputSchema["properties"]["representative_limit"]["default"] == 3


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


def test_duplicate_external_urls_tool_returns_duplicate_clusters(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Insert units with duplicate URLs
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="canonical",
            source_entity_type="insight",
            title="Canonical Report",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"canonical_url": "https://example.com/report"},
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="source",
            source_entity_type="knowledge_node",
            title="Source Reference",
            content="Content",
            metadata={"citation": {"source_url": "https://example.com/report/"}},
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unique",
            source_entity_type="insight",
            title="Unique URL",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"external_url": "https://example.com/unique"},
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    tools = asyncio.run(mcp_server.list_tools())
    assert next(tool for tool in tools if tool.name == "duplicate_external_urls")

    response = asyncio.run(mcp_server.call_tool("duplicate_external_urls", {}))
    payload = json.loads(response[0].text)

    assert len(payload) == 1
    assert payload[0]["url"] == "https://example.com/report"
    assert payload[0]["count"] == 2
    assert len(payload[0]["units"]) == 2


def test_duplicate_external_urls_tool_returns_empty_when_no_duplicates(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Insert units with unique URLs only
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unit1",
            source_entity_type="insight",
            title="First Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"url": "https://example.com/first"},
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unit2",
            source_entity_type="insight",
            title="Second Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"url": "https://example.com/second"},
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("duplicate_external_urls", {}))
    payload = json.loads(response[0].text)

    assert payload == []


def test_duplicate_external_urls_tool_respects_limit_parameter(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Create three duplicate URL groups
    for group_id in ["a", "b", "c"]:
        for unit_id in [f"{group_id}1", f"{group_id}2"]:
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=unit_id,
                    source_entity_type="insight",
                    title=f"Unit {unit_id}",
                    content="Content",
                    content_type=ContentType.INSIGHT,
                    metadata={"url": f"https://example.com/{group_id}"},
                )
            )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("duplicate_external_urls", {"limit": 2}))
    payload = json.loads(response[0].text)

    assert len(payload) == 2


def test_metadata_key_frequency_tool_returns_empty_for_empty_graph(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("metadata_key_frequency", {}))
    payload = json.loads(response[0].text)

    assert payload == []


def test_metadata_key_frequency_tool_counts_metadata_keys(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Insert units with various metadata keys
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unit1",
            source_entity_type="insight",
            title="First Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"author": "Alice", "year": 2024},
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unit2",
            source_entity_type="insight",
            title="Second Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"author": "Bob", "category": "research"},
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="unit3",
            source_entity_type="insight",
            title="Third Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"author": "Charlie"},
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("metadata_key_frequency", {}))
    payload = json.loads(response[0].text)

    # Should return metadata keys sorted by frequency (count descending, then path)
    assert len(payload) == 3
    assert payload[0]["path"] == "author"
    assert payload[0]["count"] == 3
    assert "max" in payload[0]["source_projects"]
    assert "forty_two" in payload[0]["source_projects"]


def test_metadata_key_frequency_tool_filters_by_prefix(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unit1",
            source_entity_type="insight",
            title="Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={
                "citation": {"author": "Alice", "year": 2024},
                "summary": "Summary text",
            },
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool("metadata_key_frequency", {"prefix": "citation"})
    )
    payload = json.loads(response[0].text)

    # Should only return keys starting with "citation"
    assert len(payload) == 2
    assert all(item["path"].startswith("citation") for item in payload)
    assert {item["path"] for item in payload} == {"citation.author", "citation.year"}


def test_metadata_key_frequency_tool_respects_limit(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unit1",
            source_entity_type="insight",
            title="Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"key1": "a", "key2": "b", "key3": "c"},
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("metadata_key_frequency", {"limit": 2}))
    payload = json.loads(response[0].text)

    assert len(payload) == 2


def test_metadata_key_frequency_tool_filters_by_source_project(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Insert units from different projects with shared metadata keys
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unit1",
            source_entity_type="insight",
            title="Max Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"author": "Alice", "project_specific": "max_value"},
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="unit2",
            source_entity_type="insight",
            title="42 Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"author": "Bob", "other_key": "value"},
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool("metadata_key_frequency", {"source_project": "max"})
    )
    payload = json.loads(response[0].text)

    # Should only include keys from MAX project
    assert len(payload) == 2
    author_item = next(item for item in payload if item["path"] == "author")
    # Total count includes all units, but source_projects is filtered
    assert author_item["count"] == 2  # Total across both projects
    assert author_item["source_projects"] == ["max"]  # Only showing MAX

    # project_specific should only appear in MAX
    project_item = next(item for item in payload if item["path"] == "project_specific")
    assert project_item["count"] == 1
    assert project_item["source_projects"] == ["max"]


def test_metadata_key_frequency_tool_handles_units_without_metadata(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Mix of units with and without metadata
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unit1",
            source_entity_type="insight",
            title="With Metadata",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={"author": "Alice"},
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unit2",
            source_entity_type="insight",
            title="Without Metadata",
            content="Content",
            content_type=ContentType.INSIGHT,
            metadata={},
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("metadata_key_frequency", {}))
    payload = json.loads(response[0].text)

    # Should only count the unit with metadata
    assert len(payload) == 1
    assert payload[0]["path"] == "author"
    assert payload[0]["count"] == 1


def test_edge_relation_distribution_tool_returns_empty_for_graph_without_edges(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Create units but no edges
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="unit1",
            source_entity_type="insight",
            title="Unit 1",
            content="Content",
            content_type=ContentType.INSIGHT,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("edge_relation_distribution", {}))
    payload = json.loads(response[0].text)

    assert payload == []


def test_edge_relation_distribution_tool_counts_relations(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Create units
    unit1 = KnowledgeUnit(
        source_project=SourceProject.MAX,
        source_id="unit1",
        source_entity_type="insight",
        title="Unit 1",
        content="Content",
        content_type=ContentType.INSIGHT,
    )
    unit2 = KnowledgeUnit(
        source_project=SourceProject.MAX,
        source_id="unit2",
        source_entity_type="insight",
        title="Unit 2",
        content="Content",
        content_type=ContentType.INSIGHT,
    )
    unit3 = KnowledgeUnit(
        source_project=SourceProject.MAX,
        source_id="unit3",
        source_entity_type="insight",
        title="Unit 3",
        content="Content",
        content_type=ContentType.INSIGHT,
    )
    store.insert_unit(unit1)
    store.insert_unit(unit2)
    store.insert_unit(unit3)

    # Create edges with different relations
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=unit1.id,
            to_unit_id=unit2.id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.MANUAL,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=unit1.id,
            to_unit_id=unit3.id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.MANUAL,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=unit2.id,
            to_unit_id=unit3.id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.MANUAL,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("edge_relation_distribution", {}))
    payload = json.loads(response[0].text)

    # Should return counts grouped by relation, sorted by count desc then relation
    assert len(payload) == 2
    assert payload[0]["relation"] == EdgeRelation.RELATES_TO.value
    assert payload[0]["count"] == 2
    assert payload[1]["relation"] == EdgeRelation.REFERENCES.value
    assert payload[1]["count"] == 1


def test_edge_relation_distribution_tool_handles_single_relation(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Create units
    units = []
    for i in range(4):
        unit = KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id=f"unit{i}",
            source_entity_type="insight",
            title=f"Unit {i}",
            content="Content",
            content_type=ContentType.INSIGHT,
        )
        store.insert_unit(unit)
        units.append(unit)

    # Create edges with same relation but different unit pairs
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=units[0].id,
            to_unit_id=units[1].id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.MANUAL,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=units[1].id,
            to_unit_id=units[2].id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.MANUAL,
        )
    )
    store.insert_edge(
        KnowledgeEdge(
            from_unit_id=units[2].id,
            to_unit_id=units[3].id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.MANUAL,
        )
    )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("edge_relation_distribution", {}))
    payload = json.loads(response[0].text)

    assert len(payload) == 1
    assert payload[0]["relation"] == EdgeRelation.REFERENCES.value
    assert payload[0]["count"] == 3


def test_edge_relation_distribution_tool_handles_mixed_relations(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Create units
    units = []
    for i in range(5):
        unit = KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id=f"unit{i}",
            source_entity_type="insight",
            title=f"Unit {i}",
            content="Content",
            content_type=ContentType.INSIGHT,
        )
        store.insert_unit(unit)
        units.append(unit)

    # Create edges with various relations
    relations = [
        EdgeRelation.RELATES_TO,
        EdgeRelation.RELATES_TO,
        EdgeRelation.RELATES_TO,
        EdgeRelation.REFERENCES,
        EdgeRelation.REFERENCES,
        EdgeRelation.CHALLENGES,
    ]

    for i, relation in enumerate(relations):
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=units[i % len(units)].id,
                to_unit_id=units[(i + 1) % len(units)].id,
                relation=relation,
                source=EdgeSource.MANUAL,
            )
        )
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("edge_relation_distribution", {}))
    payload = json.loads(response[0].text)

    assert len(payload) == 3
    # Sorted by count desc, then relation
    assert payload[0]["relation"] == EdgeRelation.RELATES_TO.value
    assert payload[0]["count"] == 3
    assert payload[1]["relation"] == EdgeRelation.REFERENCES.value
    assert payload[1]["count"] == 2
    assert payload[2]["relation"] == EdgeRelation.CHALLENGES.value
    assert payload[2]["count"] == 1


def test_unit_creation_rate_tool_returns_empty_for_empty_graph(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))
    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("unit_creation_rate", {}))
    payload = json.loads(response[0].text)

    assert payload == []


def test_unit_creation_rate_tool_buckets_by_month(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Create units across different months
    base_time = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)

    # January: 2 units
    for i in range(2):
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=f"jan-{i}",
                source_entity_type="insight",
                title=f"January Unit {i}",
                content="Content",
                content_type=ContentType.INSIGHT,
                created_at=base_time + timedelta(days=i),
            )
        )

    # February: 3 units
    feb_time = datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc)
    for i in range(3):
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=f"feb-{i}",
                source_entity_type="insight",
                title=f"February Unit {i}",
                content="Content",
                content_type=ContentType.INSIGHT,
                created_at=feb_time + timedelta(days=i),
            )
        )

    # March: 1 unit
    mar_time = datetime(2026, 3, 5, 10, 0, tzinfo=timezone.utc)
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="mar-0",
            source_entity_type="insight",
            title="March Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            created_at=mar_time,
        )
    )

    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("unit_creation_rate", {"bucket": "monthly"}))
    payload = json.loads(response[0].text)

    # Should return months sorted descending (newest first)
    assert len(payload) == 3
    assert payload[0]["period"] == "2026-03"
    assert payload[0]["count"] == 1
    assert payload[1]["period"] == "2026-02"
    assert payload[1]["count"] == 3
    assert payload[2]["period"] == "2026-01"
    assert payload[2]["count"] == 2


def test_unit_creation_rate_tool_buckets_by_day(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Create units across different days
    base_time = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)

    for i in range(3):
        for j in range(i + 1):  # Day 1: 1 unit, Day 2: 2 units, Day 3: 3 units
            store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"day{i}-unit{j}",
                    source_entity_type="insight",
                    title=f"Day {i} Unit {j}",
                    content="Content",
                    content_type=ContentType.INSIGHT,
                    created_at=base_time + timedelta(days=i, hours=j),
                )
            )

    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("unit_creation_rate", {"bucket": "daily"}))
    payload = json.loads(response[0].text)

    assert len(payload) == 3
    assert payload[0]["period"] == "2026-05-03"
    assert payload[0]["count"] == 3
    assert payload[1]["period"] == "2026-05-02"
    assert payload[1]["count"] == 2
    assert payload[2]["period"] == "2026-05-01"
    assert payload[2]["count"] == 1


def test_unit_creation_rate_tool_filters_by_source_project(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    base_time = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)

    # MAX project units
    for i in range(2):
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=f"max-{i}",
                source_entity_type="insight",
                title=f"MAX Unit {i}",
                content="Content",
                content_type=ContentType.INSIGHT,
                created_at=base_time + timedelta(days=i),
            )
        )

    # FORTY_TWO project units
    for i in range(3):
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id=f"42-{i}",
                source_entity_type="insight",
                title=f"42 Unit {i}",
                content="Content",
                content_type=ContentType.INSIGHT,
                created_at=base_time + timedelta(days=i),
            )
        )

    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool("unit_creation_rate", {"bucket": "monthly", "source_project": "max"})
    )
    payload = json.loads(response[0].text)

    # Should only count MAX units
    assert len(payload) == 1
    assert payload[0]["period"] == "2026-05"
    assert payload[0]["count"] == 2


def test_unit_creation_rate_tool_filters_by_date_range(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Create units across several months
    for month in [1, 2, 3, 4, 5]:
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=f"month-{month}",
                source_entity_type="insight",
                title=f"Month {month} Unit",
                content="Content",
                content_type=ContentType.INSIGHT,
                created_at=datetime(2026, month, 15, 10, 0, tzinfo=timezone.utc),
            )
        )

    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    # Filter to only February through April
    response = asyncio.run(
        mcp_server.call_tool(
            "unit_creation_rate",
            {
                "bucket": "monthly",
                "created_after": "2026-02-01T00:00:00+00:00",
                "created_before": "2026-05-01T00:00:00+00:00",
            },
        )
    )
    payload = json.loads(response[0].text)

    assert len(payload) == 3
    assert payload[0]["period"] == "2026-04"
    assert payload[1]["period"] == "2026-03"
    assert payload[2]["period"] == "2026-02"


def test_unit_creation_rate_tool_respects_limit(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Create units across 5 months
    for month in [1, 2, 3, 4, 5]:
        store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id=f"month-{month}",
                source_entity_type="insight",
                title=f"Month {month} Unit",
                content="Content",
                content_type=ContentType.INSIGHT,
                created_at=datetime(2026, month, 15, 10, 0, tzinfo=timezone.utc),
            )
        )

    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(
        mcp_server.call_tool("unit_creation_rate", {"bucket": "monthly", "limit": 3})
    )
    payload = json.loads(response[0].text)

    # Should return only 3 most recent months
    assert len(payload) == 3
    assert payload[0]["period"] == "2026-05"
    assert payload[1]["period"] == "2026-04"
    assert payload[2]["period"] == "2026-03"


def test_unit_creation_rate_tool_handles_sparse_data(tmp_path, monkeypatch):
    db_path = tmp_path / "graph.db"
    store = Store(str(db_path))

    # Create units with gaps (January and March, skip February)
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="jan",
            source_entity_type="insight",
            title="January Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            created_at=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        )
    )
    store.insert_unit(
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="mar",
            source_entity_type="insight",
            title="March Unit",
            content="Content",
            content_type=ContentType.INSIGHT,
            created_at=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
        )
    )

    store.close()

    monkeypatch.setattr(mcp_server, "_get_store", lambda: Store(str(db_path)))

    response = asyncio.run(mcp_server.call_tool("unit_creation_rate", {"bucket": "monthly"}))
    payload = json.loads(response[0].text)

    # Should only return months with data (no February)
    assert len(payload) == 2
    assert payload[0]["period"] == "2026-03"
    assert payload[1]["period"] == "2026-01"
