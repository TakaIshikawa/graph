"""MCP server exposing graph tools for LLM-native agents."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from graph.config import settings
from graph.cli.main import (
    _backlinks_payload,
    _do_context_pack,
    _do_delete_edge,
    _do_export_graphml,
    _do_export_json,
    _do_export_mermaid,
    _do_export_neighborhood,
    _do_export_obsidian,
    _do_export_queries,
    _do_export_report,
    _do_export_turtle,
    _do_embeddings_status,
    _do_stats_snapshot,
    _do_apply_tags_to_search,
    _do_import_json,
    _do_import_edges_csv,
    _do_import_queries,
    _do_infer_edges,
    _do_integrity_audit,
    _do_merge_units,
    _do_delete_unit,
    _do_pin_unit,
    _do_pinned_units,
    _do_rename_tag,
    _do_tag_graph,
    _do_unpin_unit,
    _do_update_edge,
    _do_search,
    _do_search_facets,
    _do_similar,
    _do_update_unit,
    _list_edges_payload,
    _search_filters_dict,
    _validate_search_filters,
    SEARCH_SORTS,
)
from graph.graph.service import GraphService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState

server = Server("graph")

SUPPORTED_SYNC_PROJECTS = [
    "forty_two",
    "max",
    "presence",
    "me",
    "kindle",
    "sota",
    "bookmarks",
    "csv",
    "jsonl",
    "opml",
    "text",
    "html",
]

SEARCH_FILTER_SCHEMA = {
    "source_project": {
        "type": "string",
        "enum": SUPPORTED_SYNC_PROJECTS,
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
    "created_after": {
        "type": "string",
        "description": "Filter to units created on or after an ISO-8601 date/datetime",
    },
    "created_before": {
        "type": "string",
        "description": "Filter to units created on or before an ISO-8601 date/datetime",
    },
    "updated_after": {
        "type": "string",
        "description": "Filter to units updated on or after an ISO-8601 date/datetime",
    },
    "updated_before": {
        "type": "string",
        "description": "Filter to units updated on or before an ISO-8601 date/datetime",
    },
    "min_utility": {
        "type": "number",
        "description": "Filter to units with utility score at least this value",
    },
    "max_utility": {
        "type": "number",
        "description": "Filter to units with utility score at most this value",
    },
    "min_confidence": {
        "type": "number",
        "description": "Filter to units with confidence at least this value",
    },
    "max_confidence": {
        "type": "number",
        "description": "Filter to units with confidence at most this value",
    },
}


def _get_store() -> Store:
    return Store(settings.database_url)


def _get_adapter(name: str):
    from graph.adapters.bookmarks import BookmarksAdapter
    from graph.adapters.csv_adapter import CsvAdapter
    from graph.adapters.forty_two import FortyTwoAdapter
    from graph.adapters.html import HtmlAdapter
    from graph.adapters.jsonl_adapter import JsonlAdapter
    from graph.adapters.kindle import KindleAdapter
    from graph.adapters.max_adapter import MaxAdapter
    from graph.adapters.me import MeAdapter
    from graph.adapters.opml import OpmlAdapter
    from graph.adapters.presence import PresenceAdapter
    from graph.adapters.sota import SOTAAdapter
    from graph.adapters.text import TextAdapter

    mapping = {
        "forty_two": lambda: FortyTwoAdapter(db_path=settings.forty_two_db),
        "max": lambda: MaxAdapter(db_path=settings.max_db),
        "presence": lambda: PresenceAdapter(
            db_path=settings.presence_db, min_score=settings.content_min_score
        ),
        "me": lambda: MeAdapter(config_path=settings.me_config),
        "kindle": lambda: KindleAdapter(db_path=settings.kindle_db),
        "sota": lambda: SOTAAdapter(db_path=settings.sota_db),
        "bookmarks": lambda: BookmarksAdapter(path=settings.bookmarks_path),
        "csv": lambda: CsvAdapter(path=settings.csv_path),
        "jsonl": lambda: JsonlAdapter(path=settings.jsonl_path),
        "opml": lambda: OpmlAdapter(path=settings.opml_path),
        "text": lambda: TextAdapter(root_path=settings.text_root),
        "html": lambda: HtmlAdapter(root_path=settings.html_root),
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
                        "enum": [*SUPPORTED_SYNC_PROJECTS, "all"],
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
                    "sort": {
                        "type": "string",
                        "enum": list(SEARCH_SORTS),
                        "default": "relevance",
                        "description": "Result ordering",
                    },
                    **SEARCH_FILTER_SCHEMA,
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
            name="similar_units",
            description="Find units similar to an existing unit id using stored vectors or local full-text fallback.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string", "description": "Seed unit id"},
                    "limit": {"type": "integer", "default": 10},
                    **SEARCH_FILTER_SCHEMA,
                },
                "required": ["unit_id"],
            },
        ),
        Tool(
            name="pinned_units",
            description="List pinned knowledge units newest pin first, with optional filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "include_content": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include full content in each unit payload",
                    },
                    "source_project": SEARCH_FILTER_SCHEMA["source_project"],
                    "content_type": SEARCH_FILTER_SCHEMA["content_type"],
                    "tag": SEARCH_FILTER_SCHEMA["tag"],
                },
            },
        ),
        Tool(
            name="search_facets",
            description="Summarize matching knowledge units by source, entity type, content type, and tag facets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "sort": {
                        "type": "string",
                        "enum": list(SEARCH_SORTS),
                        "default": "relevance",
                        "description": "Accepted for parity with search; facets are count summaries.",
                    },
                    **SEARCH_FILTER_SCHEMA,
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
            name="context_pack",
            description="Search knowledge units and return ranked excerpts with adjacent graph context for LLM retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10},
                    "sort": {
                        "type": "string",
                        "enum": list(SEARCH_SORTS),
                        "default": "relevance",
                        "description": "Result ordering",
                    },
                    **SEARCH_FILTER_SCHEMA,
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "fulltext"],
                        "default": "fulltext",
                        "description": "Search mode",
                    },
                    "neighbor_depth": {
                        "type": "integer",
                        "default": 1,
                        "maximum": 2,
                        "description": "Graph neighbor depth for context; capped at 2",
                    },
                    "char_budget": {
                        "type": "integer",
                        "default": 4000,
                        "description": "Total character budget for content excerpts",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="save_query",
            description="Save a reusable graph search recipe with mode, limit, and filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Saved query name"},
                    "query": {"type": "string", "description": "Search query"},
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "fulltext"],
                        "default": "fulltext",
                        "description": "Search mode",
                    },
                    "limit": {"type": "integer", "default": 10},
                    "sort": {
                        "type": "string",
                        "enum": list(SEARCH_SORTS),
                        "default": "relevance",
                        "description": "Sort order saved with the query filters",
                    },
                    **SEARCH_FILTER_SCHEMA,
                    "filters": {
                        "type": "object",
                        "description": "Structured filters such as source_project, content_type, tag, review_state, created_after, created_before, updated_after, updated_before, min_utility, max_utility",
                        "additionalProperties": True,
                        "default": {},
                    },
                },
                "required": ["name", "query"],
            },
        ),
        Tool(
            name="list_queries",
            description="List saved graph search recipes.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="export_queries",
            description="Export saved graph search recipes to a portable JSON file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination saved queries JSON path",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="import_queries",
            description="Import saved graph search recipes from a portable JSON file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Source saved queries JSON path",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="run_query",
            description="Run a saved graph search recipe.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Saved query name"},
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="delete_query",
            description="Delete a saved graph search recipe.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Saved query name"},
                },
                "required": ["name"],
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
            name="backlinks",
            description="Return incoming and outgoing references for a knowledge unit with expanded neighbor details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string", "description": "Knowledge unit ID"},
                    "direction": {
                        "type": "string",
                        "enum": ["incoming", "outgoing", "both"],
                        "default": "both",
                    },
                    "relation": {
                        "type": "string",
                        "enum": [r.value for r in EdgeRelation],
                        "description": "Filter by edge relation",
                    },
                    "limit": {"type": "integer", "default": 20},
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
            name="shortest_path",
            description=(
                "Return the shortest path between two knowledge units with ordered "
                "units and connecting edge metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "from_unit_id": {"type": "string"},
                    "to_unit_id": {"type": "string"},
                },
                "required": ["from_unit_id", "to_unit_id"],
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
            name="analyze_source_coverage",
            description="Summarize unit, edge, sync, and orphan coverage by source project and entity type.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="integrity_audit",
            description="Audit persisted graph tables for dangling edges, FTS drift, invalid JSON, and blank units.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repair_fts": {
                        "type": "boolean",
                        "default": False,
                        "description": "Restore missing FTS rows and remove stale FTS rows only",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum examples per audit category",
                    },
                },
            },
        ),
        Tool(
            name="embedding_status",
            description="Summarize embedding coverage and stale units by source project and content type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_project": SEARCH_FILTER_SCHEMA["source_project"],
                    "content_type": SEARCH_FILTER_SCHEMA["content_type"],
                    "show_stale": {
                        "type": "integer",
                        "default": 0,
                        "minimum": 0,
                        "description": "List up to this many missing or stale units needing refresh",
                    },
                },
            },
        ),
        Tool(
            name="timeline",
            description="Bucket knowledge units over time by created_at, ingested_at, or updated_at.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bucket": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "default": "month",
                    },
                    "field": {
                        "type": "string",
                        "enum": ["created_at", "ingested_at", "updated_at"],
                        "default": "created_at",
                    },
                    "start": {
                        "type": "string",
                        "description": "Inclusive ISO-8601 start date/datetime",
                    },
                    "end": {
                        "type": "string",
                        "description": "Inclusive ISO-8601 end date/datetime",
                    },
                    "limit": {"type": "integer", "description": "Max buckets to return"},
                    "source_project": {
                        "type": "string",
                        "enum": SUPPORTED_SYNC_PROJECTS,
                        "description": "Filter by source project",
                    },
                    "content_type": {
                        "type": "string",
                        "enum": [content_type.value for content_type in ContentType],
                        "description": "Filter by content type",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Require an exact graph tag",
                    },
                },
            },
        ),
        Tool(
            name="analyze_tags",
            description="Analyze graph tags with counts, source/type breakdowns, matching units, and co-occurrences.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "source_project": {
                        "type": "string",
                        "enum": SUPPORTED_SYNC_PROJECTS,
                        "description": "Filter by source project",
                    },
                    "content_type": {
                        "type": "string",
                        "enum": [
                            "insight",
                            "finding",
                            "idea",
                            "design_brief",
                            "artifact",
                            "metadata",
                        ],
                        "description": "Filter by content type",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Show detail for one exact tag",
                    },
                },
            },
        ),
        Tool(
            name="tag_graph",
            description="Return a weighted tag co-occurrence graph with node unit counts and representative unit ids for each pair.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "min_count": {
                        "type": "integer",
                        "default": 1,
                        "description": "Minimum co-occurrence count required for a tag pair",
                    },
                    "source_project": {
                        "type": "string",
                        "enum": SUPPORTED_SYNC_PROJECTS,
                        "description": "Filter by source project",
                    },
                    "content_type": {
                        "type": "string",
                        "enum": [
                            "insight",
                            "finding",
                            "idea",
                            "design_brief",
                            "artifact",
                            "metadata",
                        ],
                        "description": "Filter by content type",
                    },
                },
            },
        ),
        Tool(
            name="suggest_tag_synonyms",
            description="Suggest likely synonymous or variant tags without modifying stored data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "min_similarity": {
                        "type": "number",
                        "default": 0.8,
                        "description": "Minimum normalized character similarity",
                    },
                },
            },
        ),
        Tool(
            name="rename_tag",
            description="Rename or merge one exact tag across matching units, with optional dry run.",
            inputSchema={
                "type": "object",
                "properties": {
                    "old_tag": {
                        "type": "string",
                        "description": "Exact tag to replace",
                    },
                    "new_tag": {
                        "type": "string",
                        "description": "Replacement tag",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Preview changes without writing",
                    },
                    "source_project": {
                        "type": "string",
                        "enum": SUPPORTED_SYNC_PROJECTS,
                        "description": "Filter by source project",
                    },
                    "content_type": {
                        "type": "string",
                        "enum": [
                            "insight",
                            "finding",
                            "idea",
                            "design_brief",
                            "artifact",
                            "metadata",
                        ],
                        "description": "Filter by content type",
                    },
                },
                "required": ["old_tag", "new_tag"],
            },
        ),
        Tool(
            name="apply_tags_to_search",
            description="Add and remove tags across units matched by a graph search, with optional dry run.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "add": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                        "description": "Tags to add to matching units",
                    },
                    "remove": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                        "description": "Tags to remove from matching units",
                    },
                    "limit": {"type": "integer", "default": 10},
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "fulltext"],
                        "default": "fulltext",
                        "description": "Search mode",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Preview changes without writing",
                    },
                    **SEARCH_FILTER_SCHEMA,
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="analyze_links",
            description="Inventory external http/https links across knowledge unit content and metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Filter by exact domain",
                    },
                    "limit": {"type": "integer", "default": 20},
                },
            },
        ),
        Tool(
            name="suggest_edges",
            description="Suggest likely missing graph edges from shared tags, links, and text overlap without writing data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "min_score": {
                        "type": "number",
                        "default": 0.4,
                        "description": "Minimum suggestion score",
                    },
                    "source_project": {
                        "type": "string",
                        "enum": SUPPORTED_SYNC_PROJECTS,
                        "description": "Filter candidate units by source project",
                    },
                },
            },
        ),
        Tool(
            name="analyze_duplicates",
            description="Find likely duplicate knowledge units by repeated titles and near-identical content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "source_project": {
                        "type": "string",
                        "enum": SUPPORTED_SYNC_PROJECTS,
                        "description": "Filter by source project",
                    },
                    "content_type": {
                        "type": "string",
                        "enum": [
                            "insight",
                            "finding",
                            "idea",
                            "design_brief",
                            "artifact",
                            "metadata",
                        ],
                        "description": "Filter by content type",
                    },
                },
            },
        ),
        Tool(
            name="review_queue",
            description="Rank knowledge units worth resurfacing for review.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "source_project": {
                        "type": "string",
                        "enum": SUPPORTED_SYNC_PROJECTS,
                        "description": "Filter by source project",
                    },
                    "content_type": {
                        "type": "string",
                        "enum": [
                            "insight",
                            "finding",
                            "idea",
                            "design_brief",
                            "artifact",
                            "metadata",
                        ],
                        "description": "Filter by content type",
                    },
                },
            },
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
            name="update_unit",
            description="Update a manually maintained knowledge unit and refresh full-text search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string", "description": "Knowledge unit ID"},
                    "title": {"type": "string", "description": "Replacement title"},
                    "content": {"type": "string", "description": "Replacement content"},
                    "content_type": {
                        "type": "string",
                        "enum": [content_type.value for content_type in ContentType],
                        "description": "Replacement content type",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                        "description": "Tags to append without removing existing tags",
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True,
                        "default": {},
                        "description": "Metadata keys to merge into existing metadata",
                    },
                },
                "required": ["unit_id"],
            },
        ),
        Tool(
            name="delete_unit",
            description="Delete a knowledge unit, its full-text row, and related edges.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string", "description": "Knowledge unit ID"},
                },
                "required": ["unit_id"],
            },
        ),
        Tool(
            name="merge_units",
            description="Merge a duplicate source knowledge unit into a target, rewiring valid edges.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Source unit ID to merge and delete",
                    },
                    "target_id": {
                        "type": "string",
                        "description": "Target unit ID to keep",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Preview merge effects without writing",
                    },
                },
                "required": ["source_id", "target_id"],
            },
        ),
        Tool(
            name="pin_unit",
            description="Pin a knowledge unit by adding pin metadata and refreshing updated_at.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string", "description": "Knowledge unit ID"},
                    "reason": {
                        "type": "string",
                        "description": "Optional reason to store as pin_reason",
                    },
                },
                "required": ["unit_id"],
            },
        ),
        Tool(
            name="unpin_unit",
            description="Remove pin metadata from a knowledge unit and refresh updated_at.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string", "description": "Knowledge unit ID"},
                },
                "required": ["unit_id"],
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
            name="import_edges_csv",
            description="Import curated graph edges from a CSV file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "CSV path with from_unit_id, to_unit_id, relation, optional weight, source, metadata_json",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Validate and report outcomes without inserting edges",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="list_edges",
            description="List incoming and outgoing edges for one knowledge unit with neighbor summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string", "description": "Knowledge unit ID"},
                    "direction": {
                        "type": "string",
                        "enum": ["incoming", "outgoing", "both"],
                        "default": "both",
                    },
                    "relation": {
                        "type": "string",
                        "enum": [r.value for r in EdgeRelation],
                        "description": "Filter by edge relation",
                    },
                },
                "required": ["unit_id"],
            },
        ),
        Tool(
            name="update_edge",
            description="Update an edge by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "edge_id": {"type": "string", "description": "Edge ID"},
                    "relation": {
                        "type": "string",
                        "enum": [r.value for r in EdgeRelation],
                        "description": "Replacement edge relation",
                    },
                    "weight": {"type": "number", "description": "Replacement weight"},
                    "source": {
                        "type": "string",
                        "enum": [s.value for s in EdgeSource],
                        "description": "Replacement edge source",
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True,
                        "default": {},
                        "description": "Metadata keys to merge into existing metadata",
                    },
                },
                "required": ["edge_id"],
            },
        ),
        Tool(
            name="delete_edge",
            description="Delete an edge by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "edge_id": {"type": "string", "description": "Edge ID"},
                },
                "required": ["edge_id"],
            },
        ),
        Tool(
            name="infer_edges",
            description="Infer RELATES_TO edges between embedded units with high semantic similarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_project": {
                        "type": "string",
                        "enum": SUPPORTED_SYNC_PROJECTS,
                        "description": "Filter candidate units by source project",
                    },
                    "content_type": {
                        "type": "string",
                        "enum": [
                            "insight",
                            "finding",
                            "idea",
                            "design_brief",
                            "artifact",
                            "metadata",
                        ],
                        "description": "Filter candidate units by content type",
                    },
                    "min_similarity": {
                        "type": "number",
                        "default": 0.75,
                        "description": "Minimum cosine similarity",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 100,
                        "description": "Max candidate pairs to process",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Preview inferred edges without writing",
                    },
                },
            },
        ),
        Tool(
            name="sync_status",
            description="Inspect sync freshness for each supported source/entity pair before ingesting.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="freshness_report",
            description="Report source freshness, recent ingests, and stale sync targets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "default": 7,
                        "minimum": 0,
                        "description": "Recent ingest window and stale sync threshold in days",
                    },
                },
            },
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
                        "default": settings.obsidian_vault_path,
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
            name="export_json",
            description="Export the graph to a portable JSON backup file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination JSON backup path",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="export_graphml",
            description="Export the graph to a GraphML file for tools like Gephi and yEd.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination GraphML file path",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="export_mermaid",
            description="Export a capped graph view or focused unit neighborhood as Mermaid Markdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination Markdown file path",
                    },
                    "unit_id": {
                        "type": "string",
                        "description": "Optional center knowledge unit ID",
                    },
                    "depth": {
                        "type": "integer",
                        "default": 1,
                        "maximum": 3,
                        "description": "Traversal depth when unit_id is provided",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 100,
                        "description": "Maximum nodes to export",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="export_turtle",
            description="Export the graph to RDF Turtle for linked-data tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination Turtle file path",
                    },
                    "base_uri": {
                        "type": "string",
                        "description": "Base URI for exported knowledge unit IRIs",
                        "default": "https://graph.local/unit/",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="export_neighborhood",
            description="Export a focused JSON subgraph around one knowledge unit.",
            inputSchema={
                "type": "object",
                "properties": {
                    "unit_id": {
                        "type": "string",
                        "description": "Center knowledge unit ID",
                    },
                    "path": {
                        "type": "string",
                        "description": "Destination neighborhood JSON path",
                    },
                    "depth": {
                        "type": "integer",
                        "default": 1,
                        "maximum": 3,
                        "description": "Traversal depth from the center unit",
                    },
                },
                "required": ["unit_id", "path"],
            },
        ),
        Tool(
            name="export_report",
            description="Export a Markdown graph health report for review and sharing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination Markdown report path",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max items per report section",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="import_json",
            description="Import a portable JSON graph backup file idempotently.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Source JSON backup path",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="stats",
            description=(
                "Get a machine-readable graph statistics snapshot with unit, edge, "
                "embedding, isolation, and top-degree summaries."
            ),
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
            projects = SUPPORTED_SYNC_PROJECTS if project == "all" else [project]
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
            try:
                filters = _search_filters_dict(
                    source_project=arguments.get("source_project"),
                    content_type=arguments.get("content_type"),
                    review_state=arguments.get("review_state"),
                    tag=arguments.get("tag"),
                    created_after=arguments.get("created_after"),
                    created_before=arguments.get("created_before"),
                    updated_after=arguments.get("updated_after"),
                    updated_before=arguments.get("updated_before"),
                    min_utility=arguments.get("min_utility"),
                    max_utility=arguments.get("max_utility"),
                    min_confidence=arguments.get("min_confidence"),
                    max_confidence=arguments.get("max_confidence"),
                )
                payload = _do_search(
                    store,
                    query,
                    limit=limit,
                    mode=mode,
                    filters=filters,
                    sort=arguments.get("sort", "relevance"),
                )
            except ValueError as exc:
                payload = {
                    "error": str(exc),
                    "valid_modes": ["fulltext", "semantic", "hybrid"],
                    "valid_sorts": list(SEARCH_SORTS),
                }
                return [TextContent(type="text", text=json.dumps(payload))]
            if arguments.get("sort") is not None:
                return [TextContent(type="text", text=json.dumps(payload))]
            return [TextContent(type="text", text=json.dumps(payload["results"]))]

        elif name == "pinned_units":
            payload = _do_pinned_units(
                store,
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
                tag=arguments.get("tag"),
                limit=arguments.get("limit", 20),
                include_content=arguments.get("include_content", False),
            )
            return [TextContent(type="text", text=json.dumps(payload, default=str))]

        elif name == "similar_units":
            filters = _search_filters_dict(
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
                tag=arguments.get("tag"),
            )
            payload = _do_similar(
                store,
                arguments["unit_id"],
                limit=arguments.get("limit", 10),
                filters=filters,
            )
            return [TextContent(type="text", text=json.dumps(payload))]

        elif name == "search_facets":
            query = arguments["query"]
            try:
                filters = _search_filters_dict(
                    source_project=arguments.get("source_project"),
                    content_type=arguments.get("content_type"),
                    review_state=arguments.get("review_state"),
                    tag=arguments.get("tag"),
                    created_after=arguments.get("created_after"),
                    created_before=arguments.get("created_before"),
                    updated_after=arguments.get("updated_after"),
                    updated_before=arguments.get("updated_before"),
                    min_utility=arguments.get("min_utility"),
                    max_utility=arguments.get("max_utility"),
                    min_confidence=arguments.get("min_confidence"),
                    max_confidence=arguments.get("max_confidence"),
                )
                payload = _do_search_facets(
                    store,
                    query,
                    mode=arguments.get("mode", "fulltext"),
                    filters=filters,
                    sort=arguments.get("sort", "relevance"),
                )
            except ValueError as exc:
                payload = {
                    "error": str(exc),
                    "valid_modes": ["fulltext", "semantic", "hybrid"],
                    "valid_sorts": list(SEARCH_SORTS),
                }
            return [TextContent(type="text", text=json.dumps(payload))]

        elif name == "context_pack":
            query = arguments["query"]
            try:
                filters = _search_filters_dict(
                    source_project=arguments.get("source_project"),
                    content_type=arguments.get("content_type"),
                    review_state=arguments.get("review_state"),
                    tag=arguments.get("tag"),
                    created_after=arguments.get("created_after"),
                    created_before=arguments.get("created_before"),
                    updated_after=arguments.get("updated_after"),
                    updated_before=arguments.get("updated_before"),
                    min_utility=arguments.get("min_utility"),
                    max_utility=arguments.get("max_utility"),
                    min_confidence=arguments.get("min_confidence"),
                    max_confidence=arguments.get("max_confidence"),
                )
                payload = _do_context_pack(
                    store,
                    query,
                    limit=arguments.get("limit", 10),
                    mode=arguments.get("mode", "fulltext"),
                    filters=filters,
                    sort=arguments.get("sort", "relevance"),
                    char_budget=arguments.get("char_budget", 4000),
                    neighbor_depth=arguments.get("neighbor_depth", 1),
                )
            except ValueError as exc:
                payload = {
                    "error": str(exc),
                    "valid_modes": ["fulltext", "semantic", "hybrid"],
                    "valid_sorts": list(SEARCH_SORTS),
                }
            return [TextContent(type="text", text=json.dumps(payload))]

        elif name == "save_query":
            if arguments.get("sort", "relevance") not in SEARCH_SORTS:
                payload = {
                    "error": (
                        f"Unknown sort: {arguments.get('sort')}. "
                        f"Use one of: {', '.join(SEARCH_SORTS)}."
                    ),
                    "valid_sorts": list(SEARCH_SORTS),
                }
                return [TextContent(type="text", text=json.dumps(payload))]
            try:
                filters = dict(arguments.get("filters", {}))
                filters.update(
                    _search_filters_dict(
                        source_project=arguments.get("source_project"),
                        content_type=arguments.get("content_type"),
                        review_state=arguments.get("review_state"),
                        tag=arguments.get("tag"),
                        created_after=arguments.get("created_after"),
                        created_before=arguments.get("created_before"),
                        updated_after=arguments.get("updated_after"),
                        updated_before=arguments.get("updated_before"),
                        min_utility=arguments.get("min_utility"),
                        max_utility=arguments.get("max_utility"),
                        min_confidence=arguments.get("min_confidence"),
                        max_confidence=arguments.get("max_confidence"),
                        sort=(
                            arguments.get("sort")
                            if arguments.get("sort", "relevance") != "relevance"
                            else None
                        ),
                    )
                )
                _validate_search_filters(filters)
                saved = store.save_query(
                    name=arguments["name"],
                    query=arguments["query"],
                    mode=arguments.get("mode", "fulltext"),
                    limit=arguments.get("limit", 10),
                    filters=filters,
                )
            except ValueError as exc:
                return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]
            return [TextContent(type="text", text=json.dumps(saved))]

        elif name == "list_queries":
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"queries": store.list_saved_queries()}),
                )
            ]

        elif name == "export_queries":
            stats = _do_export_queries(store, arguments["path"])
            return [TextContent(type="text", text=json.dumps(stats))]

        elif name == "import_queries":
            try:
                stats = _do_import_queries(store, arguments["path"])
            except ValueError as exc:
                payload = {
                    "error": "import_failed",
                    "message": str(exc),
                    "path": arguments["path"],
                }
                return [TextContent(type="text", text=json.dumps(payload))]
            return [TextContent(type="text", text=json.dumps(stats))]

        elif name == "run_query":
            saved = store.get_saved_query(arguments["name"])
            if not saved:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "name": arguments["name"],
                                "found": False,
                                "error": f"Saved query not found: {arguments['name']}",
                            }
                        ),
                    )
                ]
            try:
                payload = _do_search(
                    store,
                    saved["query"],
                    limit=saved["limit"],
                    mode=saved["mode"],
                    filters=saved["filters"],
                )
            except ValueError as exc:
                payload = {
                    "error": str(exc),
                    "valid_modes": ["fulltext", "semantic", "hybrid"],
                    "valid_sorts": list(SEARCH_SORTS),
                }
            payload["saved_query"] = saved["name"]
            return [TextContent(type="text", text=json.dumps(payload))]

        elif name == "delete_query":
            deleted = store.delete_saved_query(arguments["name"])
            payload = {"name": arguments["name"], "deleted": deleted}
            if not deleted:
                payload["error"] = f"Saved query not found: {arguments['name']}"
            return [TextContent(type="text", text=json.dumps(payload))]

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

        elif name == "backlinks":
            direction = arguments.get("direction", "both")
            if direction not in ("incoming", "outgoing", "both"):
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "center": None,
                                "links": [],
                                "error": "invalid_direction",
                                "message": "direction must be incoming, outgoing, or both",
                            }
                        ),
                    )
                ]
            payload = _backlinks_payload(
                store,
                arguments["unit_id"],
                direction=direction,
                relation=arguments.get("relation"),
                limit=arguments.get("limit", 20),
            )
            return [TextContent(type="text", text=json.dumps(payload, default=str))]

        elif name == "find_path":
            gs = GraphService(store)
            gs.rebuild()
            path = gs.shortest_path(arguments["from_id"], arguments["to_id"])
            if path is None:
                return [
                    TextContent(
                        type="text", text=json.dumps({"path": None, "message": "No path found"})
                    )
                ]

            path_details = []
            for nid in path:
                n = store.get_unit(nid)
                if n:
                    path_details.append(
                        {"id": n.id, "title": n.title, "source_project": n.source_project}
                    )

            return [TextContent(type="text", text=json.dumps({"path": path_details}))]

        elif name == "shortest_path":
            gs = GraphService(store)
            gs.rebuild()
            payload = gs.build_shortest_path_payload(
                arguments["from_unit_id"],
                arguments["to_unit_id"],
            )
            return [TextContent(type="text", text=json.dumps(payload, default=str))]

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
                        nodes.append(
                            {"id": n.id, "title": n.title, "source_project": n.source_project}
                        )
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
                    results.append(
                        {
                            "unit_id": g["unit_id"],
                            "title": n.title,
                            "source_project": n.source_project,
                            "gap_type": g["gap_type"],
                            "score": g["score"],
                            "reason": g["reason"],
                        }
                    )

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
                    results.append(
                        {
                            "id": nid,
                            "title": n.title,
                            "source_project": n.source_project,
                            "pagerank": round(score, 6),
                        }
                    )

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
                    results.append(
                        {
                            "id": nid,
                            "title": n.title,
                            "source_project": n.source_project,
                            "betweenness": round(score, 6),
                        }
                    )

            return [TextContent(type="text", text=json.dumps(results))]

        elif name == "cross_project":
            gs = GraphService(store)
            gs.rebuild()
            connections = gs.cross_project_connections()
            return [TextContent(type="text", text=json.dumps(connections))]

        elif name == "analyze_source_coverage":
            gs = GraphService(store)
            result = gs.analyze_source_coverage()
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "integrity_audit":
            result = _do_integrity_audit(
                store,
                repair_fts=arguments.get("repair_fts", False),
                limit=arguments.get("limit", 20),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "embedding_status":
            result = _do_embeddings_status(
                store,
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
                show_stale=arguments.get("show_stale", 0),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "timeline":
            gs = GraphService(store)
            result = gs.analyze_timeline(
                bucket=arguments.get("bucket", "month"),
                field=arguments.get("field", "created_at"),
                start=arguments.get("start"),
                end=arguments.get("end"),
                limit=arguments.get("limit"),
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
                tag=arguments.get("tag"),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "analyze_tags":
            gs = GraphService(store)
            result = gs.analyze_tags(
                tag=arguments.get("tag"),
                limit=arguments.get("limit", 20),
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "tag_graph":
            result = _do_tag_graph(
                store,
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
                min_count=arguments.get("min_count", 1),
                limit=arguments.get("limit", 20),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "suggest_tag_synonyms":
            gs = GraphService(store)
            result = gs.suggest_tag_synonyms(
                limit=arguments.get("limit", 20),
                min_similarity=arguments.get("min_similarity", 0.8),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "rename_tag":
            result = _do_rename_tag(
                store,
                arguments["old_tag"],
                arguments["new_tag"],
                dry_run=arguments.get("dry_run", False),
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "apply_tags_to_search":
            filters = _search_filters_dict(
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
                review_state=arguments.get("review_state"),
                tag=arguments.get("tag"),
                created_after=arguments.get("created_after"),
                created_before=arguments.get("created_before"),
                updated_after=arguments.get("updated_after"),
                updated_before=arguments.get("updated_before"),
                min_utility=arguments.get("min_utility"),
                max_utility=arguments.get("max_utility"),
                min_confidence=arguments.get("min_confidence"),
                max_confidence=arguments.get("max_confidence"),
            )
            result = _do_apply_tags_to_search(
                store,
                arguments["query"],
                add_tags=arguments.get("add", arguments.get("add_tags", [])),
                remove_tags=arguments.get("remove", arguments.get("remove_tags", [])),
                limit=arguments.get("limit", 10),
                mode=arguments.get("mode", "fulltext"),
                filters=filters,
                dry_run=arguments.get("dry_run", False),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "analyze_links":
            gs = GraphService(store)
            result = gs.analyze_links(
                domain=arguments.get("domain"),
                limit=arguments.get("limit", 20),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "suggest_edges":
            gs = GraphService(store)
            result = gs.suggest_edges(
                limit=arguments.get("limit", 20),
                min_score=arguments.get("min_score", 0.4),
                source_project=arguments.get("source_project"),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "analyze_duplicates":
            gs = GraphService(store)
            result = gs.analyze_duplicates(
                limit=arguments.get("limit", 20),
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "review_queue":
            gs = GraphService(store)
            result = gs.build_review_queue(
                limit=arguments.get("limit", 20),
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "sync_status":
            statuses = []
            for source_project, source_entity_type in _supported_sync_targets():
                state = store.get_sync_state(source_project, source_entity_type)
                statuses.append(_sync_state_to_dict(source_project, source_entity_type, state))
            return [TextContent(type="text", text=json.dumps({"sync_states": statuses}))]

        elif name == "freshness_report":
            days = max(0, int(arguments.get("days", 7)))
            report = store.freshness_report(_supported_sync_targets(), days=days)
            return [TextContent(type="text", text=json.dumps({"days": days, "results": report}))]

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

        elif name == "update_unit":
            try:
                payload = _do_update_unit(
                    store,
                    arguments["unit_id"],
                    title=arguments.get("title"),
                    content=arguments.get("content"),
                    content_type=arguments.get("content_type"),
                    tags=arguments.get("tags", []),
                    metadata=arguments.get("metadata", {}),
                )
            except ValueError as exc:
                payload = {
                    "unit_id": arguments.get("unit_id"),
                    "updated": False,
                    "error": str(exc),
                }
            return [TextContent(type="text", text=json.dumps(payload, default=str))]

        elif name == "delete_unit":
            payload = _do_delete_unit(store, arguments["unit_id"])
            return [TextContent(type="text", text=json.dumps(payload))]

        elif name == "merge_units":
            try:
                payload = _do_merge_units(
                    store,
                    arguments["source_id"],
                    arguments["target_id"],
                    dry_run=arguments.get("dry_run", False),
                )
            except ValueError as exc:
                payload = {
                    "source_id": arguments.get("source_id"),
                    "target_id": arguments.get("target_id"),
                    "dry_run": arguments.get("dry_run", False),
                    "merged": False,
                    "error": str(exc),
                }
            return [TextContent(type="text", text=json.dumps(payload, default=str))]

        elif name == "pin_unit":
            payload = _do_pin_unit(
                store,
                arguments["unit_id"],
                reason=arguments.get("reason"),
            )
            return [TextContent(type="text", text=json.dumps(payload, default=str))]

        elif name == "unpin_unit":
            payload = _do_unpin_unit(store, arguments["unit_id"])
            return [TextContent(type="text", text=json.dumps(payload, default=str))]

        elif name == "add_edge":
            try:
                edge = KnowledgeEdge(
                    from_unit_id=arguments["from_unit_id"],
                    to_unit_id=arguments["to_unit_id"],
                    relation=EdgeRelation(arguments["relation"]),
                    weight=arguments.get("weight", 1.0),
                    source=EdgeSource.MANUAL,
                )
            except ValueError as exc:
                payload = {
                    "inserted": False,
                    "error": str(exc),
                    "from_unit_id": arguments.get("from_unit_id"),
                    "to_unit_id": arguments.get("to_unit_id"),
                }
                return [TextContent(type="text", text=json.dumps(payload))]
            inserted = store.insert_edge(edge)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "id": inserted.id,
                            "from_unit_id": inserted.from_unit_id,
                            "to_unit_id": inserted.to_unit_id,
                            "relation": inserted.relation,
                        }
                    ),
                )
            ]

        elif name == "import_edges_csv":
            try:
                payload = _do_import_edges_csv(
                    store,
                    arguments["path"],
                    dry_run=arguments.get("dry_run", False),
                )
            except OSError as exc:
                payload = {
                    "path": arguments.get("path"),
                    "dry_run": arguments.get("dry_run", False),
                    "inserted": 0,
                    "skipped_existing": 0,
                    "invalid": [{"row_number": None, "reasons": [str(exc)]}],
                    "inserted_rows": [],
                    "skipped_existing_rows": [],
                    "error": "file_error",
                }
            return [TextContent(type="text", text=json.dumps(payload, default=str))]

        elif name == "list_edges":
            try:
                payload = _list_edges_payload(
                    store,
                    arguments["unit_id"],
                    direction=arguments.get("direction", "both"),
                    relation=arguments.get("relation"),
                )
            except ValueError as exc:
                payload = {
                    "unit_id": arguments.get("unit_id"),
                    "edges": [],
                    "error": str(exc),
                }
            return [TextContent(type="text", text=json.dumps(payload, default=str))]

        elif name == "update_edge":
            try:
                payload = _do_update_edge(
                    store,
                    arguments["edge_id"],
                    relation=arguments.get("relation"),
                    weight=arguments.get("weight"),
                    source=arguments.get("source"),
                    metadata=arguments.get("metadata", {}),
                )
            except ValueError as exc:
                payload = {
                    "edge_id": arguments.get("edge_id"),
                    "updated": False,
                    "error": str(exc),
                }
            return [TextContent(type="text", text=json.dumps(payload, default=str))]

        elif name == "delete_edge":
            payload = _do_delete_edge(store, arguments["edge_id"])
            return [TextContent(type="text", text=json.dumps(payload))]

        elif name == "infer_edges":
            result = _do_infer_edges(
                store,
                source_project=arguments.get("source_project"),
                content_type=arguments.get("content_type"),
                min_similarity=arguments.get("min_similarity", 0.75),
                limit=arguments.get("limit", 100),
                dry_run=arguments.get("dry_run", False),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "stats":
            payload = _do_stats_snapshot(store)
            return [TextContent(type="text", text=json.dumps(payload))]

        elif name == "export_obsidian":
            vault_path = arguments.get("vault_path") or settings.obsidian_vault_path
            folder = arguments.get("folder", "Graph")
            clean = arguments.get("clean", False)
            written = _do_export_obsidian(
                store,
                vault_path=vault_path,
                folder=folder,
                clean=clean,
            )
            return [TextContent(type="text", text=json.dumps({"notes_written": written}))]

        elif name == "export_json":
            stats = _do_export_json(store, arguments["path"])
            return [TextContent(type="text", text=json.dumps(stats))]

        elif name == "export_graphml":
            stats = _do_export_graphml(store, arguments["path"])
            return [TextContent(type="text", text=json.dumps(stats))]

        elif name == "export_mermaid":
            try:
                stats = _do_export_mermaid(
                    store,
                    arguments["path"],
                    unit_id=arguments.get("unit_id"),
                    depth=arguments.get("depth", 1),
                    limit=arguments.get("limit", 100),
                )
            except ValueError as exc:
                try:
                    payload = json.loads(str(exc))
                except json.JSONDecodeError:
                    payload = {"error": "export_failed", "message": str(exc)}
                return [TextContent(type="text", text=json.dumps(payload))]
            return [TextContent(type="text", text=json.dumps(stats))]

        elif name == "export_turtle":
            stats = _do_export_turtle(
                store,
                arguments["path"],
                base_uri=arguments.get("base_uri", "https://graph.local/unit/"),
            )
            return [TextContent(type="text", text=json.dumps(stats))]

        elif name == "export_neighborhood":
            try:
                stats = _do_export_neighborhood(
                    store,
                    arguments["unit_id"],
                    arguments["path"],
                    depth=arguments.get("depth", 1),
                )
            except ValueError as exc:
                try:
                    payload = json.loads(str(exc))
                except json.JSONDecodeError:
                    payload = {"error": "export_failed", "message": str(exc)}
                return [TextContent(type="text", text=json.dumps(payload))]
            return [TextContent(type="text", text=json.dumps(stats))]

        elif name == "export_report":
            stats = _do_export_report(
                store,
                arguments["path"],
                limit=arguments.get("limit", 10),
            )
            return [TextContent(type="text", text=json.dumps(stats))]

        elif name == "import_json":
            stats = _do_import_json(store, arguments["path"])
            return [TextContent(type="text", text=json.dumps(stats))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    finally:
        store.close()


async def run_mcp_server() -> None:
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)
