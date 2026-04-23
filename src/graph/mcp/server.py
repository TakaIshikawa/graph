"""MCP server exposing graph tools for LLM-native agents."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from graph.config import settings
from graph.cli.main import _do_export_obsidian
from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState

server = Server("graph")

SUPPORTED_SYNC_PROJECTS = ["forty_two", "max", "presence", "me", "kindle"]


def _get_store() -> Store:
    return Store(settings.database_url)


def _get_adapter(name: str):
    from graph.adapters.forty_two import FortyTwoAdapter
    from graph.adapters.kindle import KindleAdapter
    from graph.adapters.max_adapter import MaxAdapter
    from graph.adapters.me import MeAdapter
    from graph.adapters.presence import PresenceAdapter

    mapping = {
        "forty_two": lambda: FortyTwoAdapter(db_path=settings.forty_two_db),
        "max": lambda: MaxAdapter(db_path=settings.max_db),
        "presence": lambda: PresenceAdapter(
            db_path=settings.presence_db, min_score=settings.content_min_score
        ),
        "me": lambda: MeAdapter(config_path=settings.me_config),
        "kindle": lambda: KindleAdapter(db_path=settings.kindle_db),
    }
    return mapping[name]()


def _supported_sync_targets() -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for project in SUPPORTED_SYNC_PROJECTS:
        adapter = _get_adapter(project)
        for entity_type in adapter.entity_types:
            targets.append((project, entity_type))
    return targets


def _sync_state_to_dict(
    source_project: str,
    source_entity_type: str,
    state: SyncState | None,
) -> dict:
    if state is None:
        return {
            "source_project": source_project,
            "source_entity_type": source_entity_type,
            "has_sync_state": False,
            "last_sync_at": None,
            "last_source_id": None,
            "items_synced": 0,
        }

    last_sync_at = state.last_sync_at
    if last_sync_at is not None and hasattr(last_sync_at, "isoformat"):
        last_sync_at = last_sync_at.isoformat()

    return {
        "source_project": source_project,
        "source_entity_type": source_entity_type,
        "has_sync_state": True,
        "last_sync_at": last_sync_at,
        "last_source_id": state.last_source_id,
        "items_synced": state.items_synced,
    }


def _unit_to_dict(unit: KnowledgeUnit) -> dict:
    review_metadata = {}
    if unit.source_project == SourceProject.MAX and unit.source_entity_type == "buildable_unit":
        review_metadata = {
            key: unit.metadata.get(key)
            for key in [
                "review_state",
                "feedback_outcome",
                "feedback_reason",
                "reviewed_at",
                "is_approved",
                "graph_labels",
                "buyer",
                "validation_plan",
                "domain",
            ]
            if unit.metadata.get(key) is not None
        }
    return {
        "id": unit.id,
        "source_id": unit.source_id,
        "source_project": unit.source_project,
        "source_entity_type": unit.source_entity_type,
        "title": unit.title,
        "content": unit.content[:500],
        "content_type": unit.content_type,
        "tags": unit.tags,
        "metadata": review_metadata,
        "confidence": unit.confidence,
        "utility_score": unit.utility_score,
        "created_at": str(unit.created_at),
    }


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="ingest",
            description="Ingest knowledge from source projects. Runs adapters to pull new/updated items.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "enum": ["forty_two", "max", "presence", "me", "kindle", "all"],
                        "description": "Source project to ingest from, or 'all'",
                    },
                    "full": {
                        "type": "boolean",
                        "default": False,
                        "description": "Ignore sync state and re-upsert all matching source items",
                    },
                },
                "required": ["project"],
            },
        ),
        Tool(
            name="search",
            description="Search knowledge units by semantic similarity and/or full-text. Returns ranked results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10},
                    "source_project": {
                        "type": "string",
                        "enum": ["forty_two", "max", "presence", "me", "kindle"],
                        "description": "Filter by source project",
                    },
                    "content_type": {
                        "type": "string",
                        "enum": ["insight", "finding", "idea", "artifact", "metadata"],
                        "description": "Filter by content type",
                    },
                    "review_state": {
                        "type": "string",
                        "enum": ["approved", "rejected", "unreviewed"],
                        "description": "Filter max ideas by review state",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Require an exact graph tag",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "fulltext"],
                        "default": "fulltext",
                        "description": "Search mode",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_unit",
            description="Get a knowledge unit by ID with its edges and metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string", "description": "Knowledge unit ID"},
                },
                "required": ["unit_id"],
            },
        ),
        Tool(
            name="traverse",
            description="Get a knowledge unit's neighborhood in the graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string"},
                    "depth": {"type": "integer", "default": 1, "maximum": 3},
                },
                "required": ["unit_id"],
            },
        ),
        Tool(
            name="find_path",
            description="Find shortest path between two knowledge units.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_id": {"type": "string"},
                    "to_id": {"type": "string"},
                },
                "required": ["from_id", "to_id"],
            },
        ),
        Tool(
            name="analyze_clusters",
            description="Find clusters of related knowledge in the graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_size": {"type": "integer", "default": 3},
                },
            },
        ),
        Tool(
            name="analyze_gaps",
            description="Identify under-connected areas and isolated knowledge.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                },
            },
        ),
        Tool(
            name="analyze_central",
            description="Find the most central/important knowledge units by PageRank.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                },
            },
        ),
        Tool(
            name="analyze_bridges",
            description="Find bridge nodes connecting different knowledge clusters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                },
            },
        ),
        Tool(
            name="cross_project",
            description="Analyze connections between different source projects.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="add_unit",
            description="Manually add a knowledge unit to the graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "content_type": {
                        "type": "string",
                        "enum": ["insight", "finding", "idea", "artifact", "metadata"],
                        "default": "insight",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["title", "content"],
            },
        ),
        Tool(
            name="add_edge",
            description="Create an edge between two knowledge units.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_unit_id": {"type": "string"},
                    "to_unit_id": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": [r.value for r in EdgeRelation],
                        "description": "Edge relation type",
                    },
                    "weight": {"type": "number", "default": 1.0},
                },
                "required": ["from_unit_id", "to_unit_id", "relation"],
            },
        ),
        Tool(
            name="sync_status",
            description="Inspect sync freshness for each supported source/entity pair before ingesting.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="export_obsidian",
            description="Export the knowledge graph to an Obsidian vault as markdown notes and index.",
            inputSchema={
                "type": "object",
                "properties": {
                    "vault_path": {
                        "type": "string",
                        "description": "Path to the Obsidian vault root",
                        "default": "/Users/taka/ObsidianVaults/note",
                    },
                    "folder": {
                        "type": "string",
                        "description": "Subfolder within the vault",
                        "default": "Graph",
                    },
                    "clean": {
                        "type": "boolean",
                        "default": False,
                        "description": "Remove the export folder before writing notes",
                    },
                },
            },
        ),
        Tool(
            name="stats",
            description="Get graph statistics: node/edge counts, density, breakdown by project and type.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    store = _get_store()

    try:
        if name == "ingest":
            project = arguments["project"]
            full = arguments.get("full", False)
            projects = (
                ["forty_two", "max", "presence", "me", "kindle"]
                if project == "all"
                else [project]
            )
            total = {"units_inserted": 0, "units_skipped": 0, "edges_inserted": 0}

            for proj in projects:
                adapter = _get_adapter(proj)
                since = None
                if not full:
                    for et in adapter.entity_types:
                        s = store.get_sync_state(proj, et)
                        if s and (since is None or s.last_sync_at < since.last_sync_at):
                            since = s

                result = adapter.ingest(since=since)
                stats = store.ingest(result, proj)

                for key in total:
                    total[key] += stats[key]

                for et in adapter.entity_types:
                    store.upsert_sync_state(
                        SyncState(
                            source_project=proj,
                            source_entity_type=et,
                            last_sync_at=datetime.now(timezone.utc),
                            items_synced=stats["units_inserted"],
                        )
                    )

            return [TextContent(type="text", text=json.dumps(total))]

        elif name == "search":
            query = arguments["query"]
            limit = arguments.get("limit", 10)
            mode = arguments.get("mode", "fulltext")
            source_project = arguments.get("source_project")
            content_type = arguments.get("content_type")
            review_state = arguments.get("review_state")
            tag = arguments.get("tag")

            if mode == "fulltext":
                fts_results = store.fts_search(query, limit=limit)
                results = []
                for r in fts_results:
                    unit = store.get_unit(r["unit_id"])
                    if unit:
                        if source_project and unit.source_project != source_project:
                            continue
                        if content_type and unit.content_type != content_type:
                            continue
                        if review_state and unit.metadata.get("review_state") != review_state:
                            continue
                        if tag and tag not in unit.tags:
                            continue
                        results.append(_unit_to_dict(unit))
                return [TextContent(type="text", text=json.dumps(results))]

            else:
                from graph.rag.embeddings import get_embedding_provider
                from graph.rag.search import RAGService

                provider = get_embedding_provider(
                    settings.embedding_provider,
                    settings.embedding_api_key,
                    settings.embedding_model,
                )
                rag = RAGService(store, provider)

                if mode == "semantic":
                    pairs = rag.search(
                        query,
                        limit=limit,
                        source_project=source_project,
                        content_type=content_type,
                        min_similarity=0.3,
                    )
                else:
                    pairs = rag.hybrid_search(query, limit=limit)

                results = []
                for unit, score in pairs:
                    if review_state and unit.metadata.get("review_state") != review_state:
                        continue
                    if tag and tag not in unit.tags:
                        continue
                    d = _unit_to_dict(unit)
                    d["score"] = round(score, 4)
                    results.append(d)
                return [TextContent(type="text", text=json.dumps(results))]

        elif name == "get_unit":
            unit = store.get_unit(arguments["unit_id"])
            if not unit:
                return [TextContent(type="text", text="Error: Unit not found")]
            edges = store.get_edges_for_unit(unit.id)
            result = _unit_to_dict(unit)
            result["content"] = unit.content  # full content
            result["metadata"] = unit.metadata
            result["edges"] = [
                {
                    "id": e.id,
                    "from_unit_id": e.from_unit_id,
                    "to_unit_id": e.to_unit_id,
                    "relation": e.relation,
                    "weight": e.weight,
                }
                for e in edges
            ]
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "traverse":
            gs = GraphService(store)
            gs.rebuild()
            depth = min(arguments.get("depth", 1), 3)
            result = gs.get_neighbors(arguments["unit_id"], depth=depth)

            if result["center"] is None:
                return [TextContent(type="text", text="Error: Unit not found in graph")]

            neighbor_details = []
            for nid in result["neighbors"]:
                n = store.get_unit(nid)
                if n:
                    neighbor_details.append(_unit_to_dict(n))

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "center": arguments["unit_id"],
                            "neighbors": neighbor_details,
                            "edges": result["edges"],
                        }
                    ),
                )
            ]

        elif name == "find_path":
            gs = GraphService(store)
            gs.rebuild()
            path = gs.shortest_path(arguments["from_id"], arguments["to_id"])
            if path is None:
                return [TextContent(type="text", text=json.dumps({"path": None, "message": "No path found"}))]

            path_details = []
            for nid in path:
                n = store.get_unit(nid)
                if n:
                    path_details.append({"id": n.id, "title": n.title, "source_project": n.source_project})

            return [TextContent(type="text", text=json.dumps({"path": path_details}))]

        elif name == "analyze_clusters":
            gs = GraphService(store)
            gs.rebuild()
            min_size = arguments.get("min_size", 3)
            found = gs.get_clusters(min_size=min_size)

            clusters = []
            for cluster in found:
                nodes = []
                for nid in cluster[:10]:
                    n = store.get_unit(nid)
                    if n:
                        nodes.append({"id": n.id, "title": n.title, "source_project": n.source_project})
                clusters.append({"size": len(cluster), "nodes": nodes})

            return [TextContent(type="text", text=json.dumps(clusters))]

        elif name == "analyze_gaps":
            gs = GraphService(store)
            gs.rebuild()
            limit = arguments.get("limit", 20)
            found = gs.find_gaps()[:limit]

            results = []
            for g in found:
                n = store.get_unit(g["unit_id"])
                if n:
                    results.append({
                        "unit_id": g["unit_id"],
                        "title": n.title,
                        "source_project": n.source_project,
                        "gap_type": g["gap_type"],
                        "score": g["score"],
                        "reason": g["reason"],
                    })

            return [TextContent(type="text", text=json.dumps(results))]

        elif name == "analyze_central":
            gs = GraphService(store)
            gs.rebuild()
            limit = arguments.get("limit", 10)
            found = gs.get_central_nodes(limit=limit)

            results = []
            for nid, score in found:
                n = store.get_unit(nid)
                if n:
                    results.append({
                        "id": nid,
                        "title": n.title,
                        "source_project": n.source_project,
                        "pagerank": round(score, 6),
                    })

            return [TextContent(type="text", text=json.dumps(results))]

        elif name == "analyze_bridges":
            gs = GraphService(store)
            gs.rebuild()
            limit = arguments.get("limit", 10)
            found = gs.get_bridges(limit=limit)

            results = []
            for nid, score in found:
                n = store.get_unit(nid)
                if n:
                    results.append({
                        "id": nid,
                        "title": n.title,
                        "source_project": n.source_project,
                        "betweenness": round(score, 6),
                    })

            return [TextContent(type="text", text=json.dumps(results))]

        elif name == "cross_project":
            gs = GraphService(store)
            gs.rebuild()
            connections = gs.cross_project_connections()
            return [TextContent(type="text", text=json.dumps(connections))]

        elif name == "sync_status":
            statuses = []
            for source_project, source_entity_type in _supported_sync_targets():
                state = store.get_sync_state(source_project, source_entity_type)
                statuses.append(
                    _sync_state_to_dict(source_project, source_entity_type, state)
                )
            return [TextContent(type="text", text=json.dumps({"sync_states": statuses}))]

        elif name == "add_unit":
            unit = KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id=f"manual-{datetime.now(timezone.utc).timestamp():.0f}",
                source_entity_type="manual",
                title=arguments["title"],
                content=arguments["content"],
                content_type=ContentType(arguments.get("content_type", "insight")),
                tags=arguments.get("tags", []),
            )
            inserted = store.insert_unit(unit)
            store.fts_index_unit(inserted)
            return [TextContent(type="text", text=json.dumps(_unit_to_dict(inserted)))]

        elif name == "add_edge":
            edge = KnowledgeEdge(
                from_unit_id=arguments["from_unit_id"],
                to_unit_id=arguments["to_unit_id"],
                relation=EdgeRelation(arguments["relation"]),
                weight=arguments.get("weight", 1.0),
                source=EdgeSource.MANUAL,
            )
            inserted = store.insert_edge(edge)
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "id": inserted.id,
                        "from_unit_id": inserted.from_unit_id,
                        "to_unit_id": inserted.to_unit_id,
                        "relation": inserted.relation,
                    }),
                )
            ]

        elif name == "stats":
            gs = GraphService(store)
            gs.rebuild()
            return [TextContent(type="text", text=json.dumps(gs.stats()))]

        elif name == "export_obsidian":
            vault_path = arguments.get("vault_path", "/Users/taka/ObsidianVaults/note")
            folder = arguments.get("folder", "Graph")
            clean = arguments.get("clean", False)
            written = _do_export_obsidian(
                store,
                vault_path=vault_path,
                folder=folder,
                clean=clean,
            )
            return [TextContent(type="text", text=json.dumps({"notes_written": written}))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    finally:
        store.close()


async def run_mcp_server() -> None:
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)
