"""CLI for the Graph personal knowledge graph."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from graph.config import settings
from graph.rag.search import SEARCH_SORTS, sort_search_results, validate_search_sort
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
from graph.types.models import SyncState

app = typer.Typer(name="graph", help="Personal Knowledge Graph — aggregate, connect, retrieve")
queries_app = typer.Typer(help="Save and run repeatable graph searches")
app.add_typer(queries_app, name="queries")
edges_app = typer.Typer(help="List and edit graph edges")
app.add_typer(edges_app, name="edges")
units_app = typer.Typer(help="Manage knowledge units")
app.add_typer(units_app, name="units")
tags_app = typer.Typer(help="Explore and mutate graph tags", invoke_without_command=True)
app.add_typer(tags_app, name="tags")

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


def _get_store() -> Store:
    return Store(settings.database_url)


def _get_adapter_for_project(name: str):
    from graph.adapters.bookmarks import BookmarksAdapter
    from graph.adapters.csv_adapter import CsvAdapter
    from graph.adapters.feed import FeedAdapter
    from graph.adapters.forty_two import FortyTwoAdapter
    from graph.adapters.html import HtmlAdapter
    from graph.adapters.jsonl_adapter import JsonlAdapter
    from graph.adapters.kindle import KindleAdapter
    from graph.adapters.markdown import MarkdownAdapter
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
        "markdown": lambda: MarkdownAdapter(root_path=settings.markdown_root),
        "kindle": lambda: KindleAdapter(db_path=settings.kindle_db),
        "sota": lambda: SOTAAdapter(db_path=settings.sota_db),
        "feed": lambda: FeedAdapter(sources=settings.feed_sources),
        "bookmarks": lambda: BookmarksAdapter(path=settings.bookmarks_path),
        "csv": lambda: CsvAdapter(path=settings.csv_path),
        "jsonl": lambda: JsonlAdapter(path=settings.jsonl_path),
        "opml": lambda: OpmlAdapter(path=settings.opml_path),
        "text": lambda: TextAdapter(root_path=settings.text_root),
        "html": lambda: HtmlAdapter(root_path=settings.html_root),
    }
    factory = mapping.get(name)
    if factory is None:
        raise typer.BadParameter(f"Unknown project: {name}. Available: {list(mapping)}")
    return factory()


def _format_unit_label(unit) -> str:
    return f"[{unit.source_project}] {unit.title}"


def _format_project_pair(projects: list[str]) -> str:
    return f"{projects[0]} <-> {projects[1]}"


def _json_echo(payload: object) -> None:
    typer.echo(json.dumps(payload, default=str, sort_keys=True))


def _supported_sync_targets() -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for project in SUPPORTED_SYNC_PROJECTS:
        adapter = _get_adapter_for_project(project)
        for entity_type in adapter.entity_types:
            targets.append((project, entity_type))
    return targets


def _do_freshness_report(store: Store, *, days: int = 7) -> dict:
    return {
        "days": max(0, days),
        "results": store.freshness_report(_supported_sync_targets(), days=days),
    }


def _do_export_json(store: Store, path: str | Path) -> dict:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = store.export_json()
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "path": str(output_path),
        "schema_version": payload["schema_version"],
        "units_exported": len(payload["units"]),
        "edges_exported": len(payload["edges"]),
    }


def _do_export_graphml(store: Store, path: str | Path) -> dict:
    from graph.graph.service import GraphService

    gs = GraphService(store)
    gs.rebuild()
    return gs.export_graphml(path)


def _do_export_mermaid(
    store: Store,
    path: str | Path,
    *,
    unit_id: str | None = None,
    depth: int = 1,
    limit: int = 100,
) -> dict:
    from graph.graph.service import GraphService

    gs = GraphService(store)
    gs.rebuild()
    return gs.export_mermaid(path, unit_id=unit_id, depth=depth, limit=limit)


def _do_export_turtle(
    store: Store, path: str | Path, *, base_uri: str = "https://graph.local/unit/"
) -> dict:
    from graph.graph.service import GraphService

    gs = GraphService(store)
    return gs.export_turtle(path, base_uri=base_uri)


def _do_export_neighborhood(
    store: Store, unit_id: str, path: str | Path, *, depth: int = 1
) -> dict:
    from graph.graph.service import GraphService

    gs = GraphService(store)
    gs.rebuild()
    return gs.export_neighborhood(unit_id, path, depth=depth)


def _do_export_report(store: Store, path: str | Path, *, limit: int = 10) -> dict:
    from graph.graph.service import GraphService

    def _line_items(items: list[str]) -> list[str]:
        return items if items else ["_None._"]

    def _unit_link(unit) -> str:
        return f"{unit.title} ({unit.source_project}/{unit.source_entity_type})"

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gs = GraphService(store)
    gs.rebuild()
    stats = gs.stats()
    central = gs.get_central_nodes(limit=limit)
    bridges = gs.get_bridges(limit=limit)
    clusters = gs.get_clusters(min_size=1)[:limit]
    gaps = gs.find_gaps()[:limit]
    tags = gs.analyze_tags(limit=limit)["tags"]
    cross_project = gs.cross_project_connections()[:limit]

    lines = [
        "# Graph Health Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Stats",
        "",
        f"- Nodes: {stats['nodes']}",
        f"- Edges: {stats['edges']}",
        f"- Components: {stats['components']}",
        f"- Density: {stats['density']}",
        "",
        "### By Project",
        "",
        *_line_items(
            [f"- {project}: {count}" for project, count in sorted(stats["by_project"].items())]
        ),
        "",
        "### By Content Type",
        "",
        *_line_items(
            [
                f"- {content_type}: {count}"
                for content_type, count in sorted(stats["by_content_type"].items())
            ]
        ),
        "",
        "## Top Central Nodes",
        "",
    ]

    central_lines = []
    for node_id, score in central:
        unit = store.get_unit(node_id)
        if unit:
            central_lines.append(f"- {_unit_link(unit)} - PageRank {score:.6f}")
    lines.extend(_line_items(central_lines))
    lines.extend(["", "## Bridge Nodes", ""])

    bridge_lines = []
    for node_id, score in bridges:
        unit = store.get_unit(node_id)
        if unit:
            bridge_lines.append(f"- {_unit_link(unit)} - Betweenness {score:.6f}")
    lines.extend(_line_items(bridge_lines))
    lines.extend(["", "## Largest Clusters", ""])

    cluster_lines = []
    for index, cluster in enumerate(clusters, 1):
        titles = []
        for node_id in cluster[:5]:
            unit = store.get_unit(node_id)
            if unit:
                titles.append(_unit_link(unit))
        suffix = f"; +{len(cluster) - 5} more" if len(cluster) > 5 else ""
        cluster_lines.append(
            f"- Cluster {index}: {len(cluster)} nodes - {', '.join(titles) or 'No units'}{suffix}"
        )
    lines.extend(_line_items(cluster_lines))
    lines.extend(["", "## Gap Candidates", ""])

    gap_lines = []
    for gap in gaps:
        unit = store.get_unit(gap["unit_id"])
        if unit:
            gap_lines.append(
                f"- [{gap['gap_type']}] {_unit_link(unit)} - "
                f"Score {gap['score']:.2f}; {gap['reason']}"
            )
    lines.extend(_line_items(gap_lines))
    lines.extend(["", "## Top Tags", ""])

    tag_lines = []
    for item in tags:
        projects = ", ".join(
            f"{project}:{count}" for project, count in item["source_projects"].items()
        )
        content_types = ", ".join(
            f"{content_type}:{count}" for content_type, count in item["content_types"].items()
        )
        tag_lines.append(
            f"- {item['tag']}: {item['count']} units "
            f"(projects: {projects or '-'}; content types: {content_types or '-'})"
        )
    lines.extend(_line_items(tag_lines))
    lines.extend(["", "## Cross-Project Connections", ""])

    cross_project_lines = [
        f"- {_format_project_pair(item['projects'])}: {item['edge_count']} edges"
        for item in cross_project
    ]
    lines.extend(_line_items(cross_project_lines))
    lines.append("")

    output_path.write_text("\n".join(lines))
    return {
        "path": str(output_path),
        "section_counts": {
            "stats": 1,
            "central": len(central_lines),
            "bridges": len(bridge_lines),
            "clusters": len(cluster_lines),
            "gaps": len(gap_lines),
            "tags": len(tag_lines),
            "cross_project": len(cross_project_lines),
        },
    }


def _anki_tsv_field(value: object) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("\t", " ").split())


def _anki_tag(tag: str) -> str:
    return "_".join(_anki_tsv_field(tag).split())


def _do_export_anki(
    store: Store,
    path: str | Path,
    *,
    source_project: str | None = None,
    content_type: str | None = None,
    tag: str | None = None,
    limit: int | None = None,
    include_tags: bool = False,
) -> dict:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[list[str]] = []
    for unit in store.get_all_units(limit=1000000000):
        if source_project and unit.source_project != source_project:
            continue
        if content_type and unit.content_type != content_type:
            continue
        if tag and tag not in unit.tags:
            continue
        if limit is not None and len(rows) >= max(0, limit):
            break

        source = (
            f"Source: {unit.source_project}/{unit.source_entity_type}/{unit.source_id} "
            f"(graph_id: {unit.id})"
        )
        back = f"{unit.content}\n\n{source}"
        tags = " ".join(_anki_tag(found_tag) for found_tag in unit.tags) if include_tags else ""
        rows.append(
            [
                _anki_tsv_field(unit.title),
                _anki_tsv_field(back),
                _anki_tsv_field(tags),
                _anki_tsv_field(unit.source_id),
            ]
        )

    output_path.write_text(
        "\n".join("\t".join(row) for row in rows) + ("\n" if rows else "")
    )
    return {
        "path": str(output_path),
        "rows_exported": len(rows),
        "include_tags": include_tags,
        "filters": {
            "source_project": source_project,
            "content_type": content_type,
            "tag": tag,
            "limit": limit,
        },
    }


def _do_import_json(store: Store, path: str | Path) -> dict:
    input_path = Path(path)
    payload = json.loads(input_path.read_text())
    stats = store.import_json(payload)
    return {"path": str(input_path), **stats}


def _do_export_queries(store: Store, path: str | Path) -> dict:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = store.export_saved_queries()
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "path": str(output_path),
        "schema_version": payload["schema_version"],
        "exported": len(payload["queries"]),
    }


def _do_import_queries(store: Store, path: str | Path) -> dict:
    input_path = Path(path)
    payload = json.loads(input_path.read_text())
    stats = store.import_saved_queries(payload)
    return {"path": str(input_path), **stats}


def _parse_metadata_json(value: str | dict | None) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--metadata-json must be a valid JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--metadata-json must be a valid JSON object.")
    return parsed


def _do_update_unit(
    store: Store,
    unit_id: str,
    *,
    title: str | None = None,
    content: str | None = None,
    content_type: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | str | None = None,
) -> dict:
    if content_type is not None:
        content_type = ContentType(content_type).value
    metadata_updates = _parse_metadata_json(metadata)
    updated = store.update_unit_fields(
        unit_id,
        title=title,
        content=content,
        content_type=content_type,
        tags=tags,
        metadata=metadata_updates,
    )
    if updated is None:
        return {
            "unit_id": unit_id,
            "updated": False,
            "error": "unit_not_found",
            "message": f"Unit not found: {unit_id}",
        }
    return {"unit_id": unit_id, "updated": True, "unit": _unit_to_json(updated)}


def _do_pin_unit(store: Store, unit_id: str, *, reason: str | None = None) -> dict:
    updated = store.pin_unit(unit_id, reason=reason)
    if updated is None:
        return {
            "unit_id": unit_id,
            "updated": False,
            "error": "unit_not_found",
            "message": f"Unit not found: {unit_id}",
        }
    return {"unit_id": unit_id, "updated": True, "unit": _unit_to_json(updated)}


def _do_unpin_unit(store: Store, unit_id: str) -> dict:
    updated = store.unpin_unit(unit_id)
    if updated is None:
        return {
            "unit_id": unit_id,
            "updated": False,
            "error": "unit_not_found",
            "message": f"Unit not found: {unit_id}",
        }
    return {"unit_id": unit_id, "updated": True, "unit": _unit_to_json(updated)}


def _do_pinned_units(
    store: Store,
    *,
    source_project: str | None = None,
    content_type: str | None = None,
    tag: str | None = None,
    limit: int | None = None,
    include_content: bool = False,
) -> dict:
    if content_type is not None:
        content_type = ContentType(content_type).value

    units = store.get_pinned_units(
        source_project=source_project,
        content_type=content_type,
        tag=tag,
        limit=limit,
    )
    return {
        "units": [_pinned_unit_to_json(unit, include_content=include_content) for unit in units],
        "count": len(units),
        "filters": {
            "source_project": source_project,
            "content_type": content_type,
            "tag": tag,
            "limit": limit,
        },
    }


def _do_rename_tag(
    store: Store,
    old_tag: str,
    new_tag: str,
    *,
    dry_run: bool = False,
    source_project: str | None = None,
    content_type: str | None = None,
) -> dict:
    if content_type is not None:
        content_type = ContentType(content_type).value
    from graph.graph.service import GraphService

    gs = GraphService(store)
    return gs.rename_tag(
        old_tag,
        new_tag,
        dry_run=dry_run,
        source_project=source_project,
        content_type=content_type,
    )


def _do_tag_graph(
    store: Store,
    *,
    source_project: str | None = None,
    content_type: str | None = None,
    min_count: int = 1,
    limit: int = 20,
) -> dict:
    if content_type is not None:
        content_type = ContentType(content_type).value
    from graph.graph.service import GraphService

    gs = GraphService(store)
    return gs.tag_graph(
        source_project=source_project,
        content_type=content_type,
        min_count=min_count,
        limit=limit,
    )


def _do_delete_unit(store: Store, unit_id: str) -> dict:
    stats = store.delete_unit(unit_id)
    if not stats["deleted"]:
        stats["error"] = "unit_not_found"
        stats["message"] = f"Unit not found: {unit_id}"
    return stats


def _do_merge_units(
    store: Store,
    source_id: str,
    target_id: str,
    *,
    dry_run: bool = False,
) -> dict:
    return store.merge_units(source_id, target_id, dry_run=dry_run)


def _edge_summary_payload(store: Store, edge, center_unit_id: str | None = None) -> dict:
    direction = None
    neighbor_id = None
    if center_unit_id is not None:
        if edge.from_unit_id == center_unit_id:
            direction = "outgoing"
            neighbor_id = edge.to_unit_id
        elif edge.to_unit_id == center_unit_id:
            direction = "incoming"
            neighbor_id = edge.from_unit_id

    payload = _knowledge_edge_to_json(edge)
    if direction is not None:
        payload["direction"] = direction
    if neighbor_id is not None:
        neighbor = store.get_unit(neighbor_id)
        payload["neighbor"] = _unit_to_json(neighbor, include_content=False) if neighbor else None
    return payload


def _list_edges_payload(
    store: Store,
    unit_id: str,
    *,
    direction: str = "both",
    relation: str | None = None,
) -> dict:
    center = store.get_unit(unit_id)
    if center is None:
        return {
            "unit_id": unit_id,
            "center": None,
            "edges": [],
            "error": "unit_not_found",
            "message": f"Unit not found: {unit_id}",
        }

    if direction not in ("incoming", "outgoing", "both"):
        raise ValueError("direction must be incoming, outgoing, or both")
    if relation is not None:
        relation = EdgeRelation(relation).value

    edges = []
    for edge in store.get_edges_for_unit(unit_id):
        edge_direction = "outgoing" if edge.from_unit_id == unit_id else "incoming"
        if direction != "both" and edge_direction != direction:
            continue
        if relation is not None and str(edge.relation) != relation:
            continue
        edges.append(_edge_summary_payload(store, edge, center_unit_id=unit_id))

    return {
        "unit_id": unit_id,
        "center": _unit_to_json(center, include_content=False),
        "edges": edges,
        "direction": direction,
        "relation": relation,
    }


def _do_update_edge(
    store: Store,
    edge_id: str,
    *,
    relation: str | None = None,
    weight: float | None = None,
    source: str | None = None,
    metadata: dict | str | None = None,
) -> dict:
    if relation is not None:
        relation = EdgeRelation(relation).value
    if source is not None:
        source = EdgeSource(source).value
    metadata_updates = _parse_metadata_json(metadata)
    updated = store.update_edge_fields(
        edge_id,
        relation=relation,
        weight=weight,
        source=source,
        metadata=metadata_updates,
    )
    if updated is None:
        return {
            "edge_id": edge_id,
            "updated": False,
            "error": "edge_not_found",
            "message": f"Edge not found: {edge_id}",
        }
    return {"edge_id": edge_id, "updated": True, "edge": _knowledge_edge_to_json(updated)}


def _do_delete_edge(store: Store, edge_id: str) -> dict:
    stats = store.delete_edge(edge_id)
    if not stats["deleted"]:
        stats["error"] = "edge_not_found"
        stats["message"] = f"Edge not found: {edge_id}"
    return stats


def _do_import_edges_csv(store: Store, path: str | Path, *, dry_run: bool = False) -> dict:
    return store.import_edges_csv(path, dry_run=dry_run)


def _content_excerpt(content: str, *, length: int = 160) -> str:
    text = " ".join((content or "").split())
    if len(text) <= length:
        return text
    return text[:length].rstrip() + "..."


def _unit_to_json(
    unit,
    *,
    score: float | None = None,
    snippet: str | None = None,
    include_content: bool = True,
) -> dict:
    created_at = (
        unit.created_at.isoformat()
        if isinstance(unit.created_at, datetime)
        else str(unit.created_at)
    )
    updated_at = (
        unit.updated_at.isoformat()
        if isinstance(unit.updated_at, datetime)
        else str(unit.updated_at)
    )
    data = {
        "id": unit.id,
        "source_project": str(unit.source_project),
        "source_id": unit.source_id,
        "source_entity_type": unit.source_entity_type,
        "title": unit.title,
        "content_type": str(unit.content_type),
        "tags": unit.tags,
        "metadata": unit.metadata,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    if include_content:
        data["content"] = unit.content
    if unit.confidence is not None:
        data["confidence"] = unit.confidence
    if unit.utility_score is not None:
        data["utility_score"] = unit.utility_score
    if score is not None:
        data["score"] = score
    if snippet is not None:
        data["snippet"] = snippet
    return data


def _pinned_unit_to_json(unit, *, include_content: bool = False) -> dict:
    data = _unit_to_json(unit, include_content=include_content)
    data["pin_reason"] = unit.metadata.get("pin_reason")
    data["pinned_at"] = unit.metadata.get("pinned_at")
    return data


def _parse_datetime_filter(value: str | datetime | None, *, name: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 date or datetime.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _unit_datetime(unit, field: str) -> datetime:
    value = getattr(unit, field)
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _validate_date_range(
    after: str | datetime | None,
    before: str | datetime | None,
    *,
    after_name: str,
    before_name: str,
) -> None:
    parsed_after = _parse_datetime_filter(after, name=after_name)
    parsed_before = _parse_datetime_filter(before, name=before_name)
    if parsed_after and parsed_before and parsed_after > parsed_before:
        raise ValueError(f"{after_name} must be on or before {before_name}.")


def _validate_search_filters(filters: dict) -> None:
    _validate_date_range(
        filters.get("created_after"),
        filters.get("created_before"),
        after_name="created_after",
        before_name="created_before",
    )
    _validate_date_range(
        filters.get("updated_after"),
        filters.get("updated_before"),
        after_name="updated_after",
        before_name="updated_before",
    )


def _edge_to_json(edge: dict) -> dict:
    return {
        "id": edge.get("id"),
        "from_unit_id": edge.get("from"),
        "to_unit_id": edge.get("to"),
        "relation": str(edge.get("relation")),
        "weight": edge.get("weight"),
    }


def _knowledge_edge_to_json(edge) -> dict:
    return {
        "id": edge.id,
        "from_unit_id": edge.from_unit_id,
        "to_unit_id": edge.to_unit_id,
        "relation": str(edge.relation),
        "weight": edge.weight,
        "source": str(edge.source),
        "metadata": edge.metadata,
        "created_at": edge.created_at,
    }


def _backlinks_payload(
    store: Store,
    unit_id: str,
    *,
    direction: str = "both",
    relation: str | None = None,
    limit: int = 20,
) -> dict:
    result = store.get_backlinks(
        unit_id,
        direction=direction,
        relation=relation,
        limit=limit,
    )
    if result["center"] is None:
        return {
            "center": None,
            "links": [],
            "direction": direction,
            "relation": relation,
            "limit": limit,
            "error": "unit_not_found",
            "message": f"Unit not found: {unit_id}",
        }

    return {
        "center": _unit_to_json(result["center"], include_content=False),
        "links": [
            {
                "direction": link["direction"],
                "relation": link["relation"],
                "edge": _knowledge_edge_to_json(link["edge"]),
                "unit": _unit_to_json(link["unit"], include_content=False),
            }
            for link in result["links"]
        ],
        "direction": direction,
        "relation": relation,
        "limit": limit,
    }


def _unit_matches_search_filters(
    unit,
    *,
    source_project: str | None = None,
    content_type: str | None = None,
    tag: str | None = None,
    review_state: str | None = None,
    created_after: str | datetime | None = None,
    created_before: str | datetime | None = None,
    updated_after: str | datetime | None = None,
    updated_before: str | datetime | None = None,
    min_utility: float | str | None = None,
    max_utility: float | str | None = None,
    min_confidence: float | str | None = None,
    max_confidence: float | str | None = None,
) -> bool:
    if source_project and str(unit.source_project) != source_project:
        return False
    if content_type and str(unit.content_type) != content_type:
        return False
    if tag and tag not in unit.tags:
        return False
    if review_state and unit.metadata.get("review_state") != review_state:
        return False
    created_after_dt = _parse_datetime_filter(created_after, name="created_after")
    created_before_dt = _parse_datetime_filter(created_before, name="created_before")
    if created_after_dt or created_before_dt:
        created_at = _unit_datetime(unit, "created_at")
        if created_after_dt and created_at < created_after_dt:
            return False
        if created_before_dt and created_at > created_before_dt:
            return False
    updated_after_dt = _parse_datetime_filter(updated_after, name="updated_after")
    updated_before_dt = _parse_datetime_filter(updated_before, name="updated_before")
    if updated_after_dt or updated_before_dt:
        updated_at = _unit_datetime(unit, "updated_at")
        if updated_after_dt and updated_at < updated_after_dt:
            return False
        if updated_before_dt and updated_at > updated_before_dt:
            return False
    if min_utility is not None or max_utility is not None:
        if unit.utility_score is None:
            return False
        utility_score = float(unit.utility_score)
        if min_utility is not None and utility_score < float(min_utility):
            return False
        if max_utility is not None and utility_score > float(max_utility):
            return False
    if min_confidence is not None or max_confidence is not None:
        if unit.confidence is None:
            return False
        confidence = float(unit.confidence)
        if min_confidence is not None and confidence < float(min_confidence):
            return False
        if max_confidence is not None and confidence > float(max_confidence):
            return False
    return True


def _search_fulltext_with_filters(
    store: Store,
    query: str,
    *,
    limit: int,
    source_project: str | None = None,
    content_type: str | None = None,
    tag: str | None = None,
    review_state: str | None = None,
    created_after: str | datetime | None = None,
    created_before: str | datetime | None = None,
    updated_after: str | datetime | None = None,
    updated_before: str | datetime | None = None,
    min_utility: float | str | None = None,
    max_utility: float | str | None = None,
    min_confidence: float | str | None = None,
    max_confidence: float | str | None = None,
    sort: str = "relevance",
) -> list[tuple[object, str]]:
    validate_search_sort(sort)
    fetch_limit = max(limit, 20)
    filtered = []

    while True:
        results = store.fts_search(
            query,
            limit=fetch_limit,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
        )
        filtered = []
        for r in results:
            unit = store.get_unit(r["unit_id"])
            if unit and _unit_matches_search_filters(
                unit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                review_state=review_state,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                min_utility=min_utility,
                max_utility=max_utility,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
            ):
                filtered.append((unit, r.get("snippet") or _content_excerpt(unit.content)))
                if sort == "relevance" and len(filtered) >= limit:
                    return filtered

        if len(results) < fetch_limit:
            return sort_search_results(filtered, sort)[:limit]

        fetch_limit *= 2


def _search_scored_with_filters(
    fetch_results,
    query: str,
    *,
    limit: int,
    source_project: str | None = None,
    content_type: str | None = None,
    tag: str | None = None,
    review_state: str | None = None,
    created_after: str | datetime | None = None,
    created_before: str | datetime | None = None,
    updated_after: str | datetime | None = None,
    updated_before: str | datetime | None = None,
    min_utility: float | str | None = None,
    max_utility: float | str | None = None,
    min_confidence: float | str | None = None,
    max_confidence: float | str | None = None,
    sort: str = "relevance",
) -> list[tuple[object, float]]:
    validate_search_sort(sort)
    fetch_limit = max(limit, 20)
    filtered = []

    while True:
        pairs = fetch_results(query, fetch_limit)
        filtered = []
        for unit, score in pairs:
            if _unit_matches_search_filters(
                unit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                review_state=review_state,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                min_utility=min_utility,
                max_utility=max_utility,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
            ):
                filtered.append((unit, score))
                if sort == "relevance" and len(filtered) >= limit:
                    return filtered

        if len(pairs) < fetch_limit:
            return sort_search_results(filtered, sort)[:limit]

        fetch_limit *= 2


def _search_filters_dict(
    *,
    source_project: str | None = None,
    content_type: str | None = None,
    tag: str | None = None,
    review_state: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    min_utility: float | None = None,
    max_utility: float | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    sort: str | None = None,
) -> dict:
    filters = {
        key: value
        for key, value in {
            "source_project": source_project,
            "content_type": content_type,
            "tag": tag,
            "review_state": review_state,
            "created_after": created_after,
            "created_before": created_before,
            "updated_after": updated_after,
            "updated_before": updated_before,
            "min_utility": min_utility,
            "max_utility": max_utility,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "sort": sort,
        }.items()
        if value is not None
    }
    _validate_search_filters(filters)
    return filters


def _validate_search_mode(mode: str) -> None:
    if mode not in ("fulltext", "semantic", "hybrid"):
        raise ValueError(f"Unknown mode: {mode}. Use fulltext, semantic, or hybrid.")


def _do_search(
    store: Store,
    query: str,
    *,
    limit: int = 10,
    mode: str = "fulltext",
    filters: dict | None = None,
    sort: str = "relevance",
) -> dict:
    _validate_search_mode(mode)
    filters = filters or {}
    _validate_search_filters(filters)
    sort = str(filters.get("sort", sort))
    validate_search_sort(sort)

    if mode == "fulltext":
        results = _search_fulltext_with_filters(
            store,
            query,
            limit=limit,
            source_project=filters.get("source_project"),
            content_type=filters.get("content_type"),
            tag=filters.get("tag"),
            review_state=filters.get("review_state"),
            created_after=filters.get("created_after"),
            created_before=filters.get("created_before"),
            updated_after=filters.get("updated_after"),
            updated_before=filters.get("updated_before"),
            min_utility=filters.get("min_utility"),
            max_utility=filters.get("max_utility"),
            min_confidence=filters.get("min_confidence"),
            max_confidence=filters.get("max_confidence"),
            sort=sort,
        )
        payload = {
            "query": query,
            "mode": mode,
            "sort": sort,
            "results": [_unit_to_json(unit, snippet=snippet) for unit, snippet in results],
            "metadata": {"sort": sort},
        }
        if filters:
            payload["filters"] = filters
        return payload

    from graph.rag.embeddings import get_embedding_provider
    from graph.rag.search import RAGService

    provider = get_embedding_provider(
        settings.embedding_provider,
        settings.embedding_api_key,
        settings.embedding_model,
    )
    rag = RAGService(store, provider)

    if mode == "semantic":
        pairs = _search_scored_with_filters(
            lambda q, fetch_limit: rag.search(
                q,
                limit=fetch_limit,
                min_similarity=0.3,
                created_after=filters.get("created_after"),
                created_before=filters.get("created_before"),
                updated_after=filters.get("updated_after"),
                updated_before=filters.get("updated_before"),
            ),
            query,
            limit=limit,
            source_project=filters.get("source_project"),
            content_type=filters.get("content_type"),
            tag=filters.get("tag"),
            review_state=filters.get("review_state"),
            created_after=filters.get("created_after"),
            created_before=filters.get("created_before"),
            updated_after=filters.get("updated_after"),
            updated_before=filters.get("updated_before"),
            min_utility=filters.get("min_utility"),
            max_utility=filters.get("max_utility"),
            min_confidence=filters.get("min_confidence"),
            max_confidence=filters.get("max_confidence"),
            sort=sort,
        )
        snippets = {unit.id: _content_excerpt(unit.content) for unit, _score in pairs}
    else:
        pairs = _search_scored_with_filters(
            lambda q, fetch_limit: rag.hybrid_search(
                q,
                limit=fetch_limit,
                created_after=filters.get("created_after"),
                created_before=filters.get("created_before"),
                updated_after=filters.get("updated_after"),
                updated_before=filters.get("updated_before"),
            ),
            query,
            limit=limit,
            source_project=filters.get("source_project"),
            content_type=filters.get("content_type"),
            tag=filters.get("tag"),
            review_state=filters.get("review_state"),
            created_after=filters.get("created_after"),
            created_before=filters.get("created_before"),
            updated_after=filters.get("updated_after"),
            updated_before=filters.get("updated_before"),
            min_utility=filters.get("min_utility"),
            max_utility=filters.get("max_utility"),
            min_confidence=filters.get("min_confidence"),
            max_confidence=filters.get("max_confidence"),
            sort=sort,
        )
        fts_snippets = {
            row["unit_id"]: row.get("snippet") or ""
            for row in store.fts_search(
                query,
                limit=max(limit * 2, 20),
                created_after=filters.get("created_after"),
                created_before=filters.get("created_before"),
                updated_after=filters.get("updated_after"),
                updated_before=filters.get("updated_before"),
            )
        }
        snippets = {
            unit.id: fts_snippets.get(unit.id) or _content_excerpt(unit.content)
            for unit, _score in pairs
        }

    payload = {
        "query": query,
        "mode": mode,
        "sort": sort,
        "results": [
            _unit_to_json(unit, score=score, snippet=snippets.get(unit.id)) for unit, score in pairs
        ],
        "metadata": {"sort": sort},
    }
    if filters:
        payload["filters"] = filters
    return payload


def _do_apply_tags_to_search(
    store: Store,
    query: str,
    *,
    add_tags: list[str] | None = None,
    remove_tags: list[str] | None = None,
    limit: int = 10,
    mode: str = "fulltext",
    filters: dict | None = None,
    dry_run: bool = False,
) -> dict:
    search_payload = _do_search(
        store,
        query,
        limit=limit,
        mode=mode,
        filters=filters,
    )
    unit_ids = [result["id"] for result in search_payload["results"]]
    result = store.apply_tags_to_units(
        unit_ids,
        add_tags=add_tags,
        remove_tags=remove_tags,
        dry_run=dry_run,
    )
    result.update(
        {
            "query": query,
            "mode": mode,
            "limit": limit,
            "filters": filters or {},
        }
    )
    return result


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _facet_payload_from_units(
    query: str,
    mode: str,
    units: list[object],
    *,
    filters: dict | None = None,
    sort: str = "relevance",
) -> dict:
    facets = {
        "source_project": {},
        "source_entity_type": {},
        "content_type": {},
        "tag": {},
    }

    for unit in units:
        values = {
            "source_project": str(unit.source_project),
            "source_entity_type": str(unit.source_entity_type),
            "content_type": str(unit.content_type),
        }
        for facet_name, value in values.items():
            counts = facets[facet_name]
            counts[value] = counts.get(value, 0) + 1
        for tag in unit.tags:
            counts = facets["tag"]
            counts[tag] = counts.get(tag, 0) + 1

    payload = {
        "query": query,
        "mode": mode,
        "sort": sort,
        "total_matches": len(units),
        "facets": {facet_name: _sorted_counts(counts) for facet_name, counts in facets.items()},
        "metadata": {"sort": sort},
    }
    if filters:
        payload["filters"] = filters
    return payload


def _do_search_facets(
    store: Store,
    query: str,
    *,
    mode: str = "fulltext",
    filters: dict | None = None,
    sort: str = "relevance",
) -> dict:
    _validate_search_mode(mode)
    filters = filters or {}
    _validate_search_filters(filters)
    sort = str(filters.get("sort", sort))
    validate_search_sort(sort)
    seen: set[str] = set()
    units = []

    def add_unit(unit) -> None:
        if unit.id in seen:
            return
        seen.add(unit.id)
        units.append(unit)

    if mode == "fulltext":
        fetch_limit = 100
        while True:
            results = store.fts_search(
                query,
                limit=fetch_limit,
                created_after=filters.get("created_after"),
                created_before=filters.get("created_before"),
                updated_after=filters.get("updated_after"),
                updated_before=filters.get("updated_before"),
            )
            for row in results:
                unit = store.get_unit(row["unit_id"])
                if unit and _unit_matches_search_filters(
                    unit,
                    source_project=filters.get("source_project"),
                    content_type=filters.get("content_type"),
                    tag=filters.get("tag"),
                    review_state=filters.get("review_state"),
                    created_after=filters.get("created_after"),
                    created_before=filters.get("created_before"),
                    updated_after=filters.get("updated_after"),
                    updated_before=filters.get("updated_before"),
                    min_utility=filters.get("min_utility"),
                    max_utility=filters.get("max_utility"),
                    min_confidence=filters.get("min_confidence"),
                    max_confidence=filters.get("max_confidence"),
                ):
                    add_unit(unit)
            if len(results) < fetch_limit:
                break
            fetch_limit *= 2
        return _facet_payload_from_units(query, mode, units, filters=filters, sort=sort)

    from graph.rag.embeddings import get_embedding_provider
    from graph.rag.search import RAGService

    provider = get_embedding_provider(
        settings.embedding_provider,
        settings.embedding_api_key,
        settings.embedding_model,
    )
    rag = RAGService(store, provider)
    if mode == "semantic":
        fetch_results = lambda q, fetch_limit: rag.search(  # noqa: E731
            q,
            limit=fetch_limit,
            min_similarity=0.3,
            created_after=filters.get("created_after"),
            created_before=filters.get("created_before"),
            updated_after=filters.get("updated_after"),
            updated_before=filters.get("updated_before"),
        )
    else:
        fetch_results = lambda q, fetch_limit: rag.hybrid_search(  # noqa: E731
            q,
            limit=fetch_limit,
            created_after=filters.get("created_after"),
            created_before=filters.get("created_before"),
            updated_after=filters.get("updated_after"),
            updated_before=filters.get("updated_before"),
        )

    fetch_limit = 100
    while True:
        pairs = fetch_results(query, fetch_limit)
        for unit, _score in pairs:
            if _unit_matches_search_filters(
                unit,
                source_project=filters.get("source_project"),
                content_type=filters.get("content_type"),
                tag=filters.get("tag"),
                review_state=filters.get("review_state"),
                created_after=filters.get("created_after"),
                created_before=filters.get("created_before"),
                updated_after=filters.get("updated_after"),
                updated_before=filters.get("updated_before"),
                min_utility=filters.get("min_utility"),
                max_utility=filters.get("max_utility"),
                min_confidence=filters.get("min_confidence"),
                max_confidence=filters.get("max_confidence"),
            ):
                add_unit(unit)
        if len(pairs) < fetch_limit:
            break
        fetch_limit *= 2

    return _facet_payload_from_units(query, mode, units, filters=filters, sort=sort)


def _do_context_pack(
    store: Store,
    query: str,
    *,
    limit: int = 10,
    mode: str = "fulltext",
    filters: dict | None = None,
    sort: str = "relevance",
    char_budget: int = 4000,
    neighbor_depth: int = 1,
) -> dict:
    from graph.rag.search import RAGService

    search_payload = _do_search(
        store,
        query,
        limit=limit,
        mode=mode,
        filters=filters,
        sort=sort,
    )
    rag = RAGService(store, provider=None)
    return rag.context_pack(
        search_payload,
        char_budget=char_budget,
        neighbor_depth=neighbor_depth,
    )


def _similar_payload_to_json(payload: dict) -> dict:
    seed = (
        _unit_to_json(payload["seed"], include_content=False)
        if payload.get("seed") is not None
        else None
    )
    return {
        key: value
        for key, value in {
            "seed_id": payload.get("seed_id"),
            "seed": seed,
            "query": payload.get("query"),
            "source_mode": payload.get("source_mode"),
            "filters": payload.get("filters", {}),
            "results": [
                {
                    **_unit_to_json(
                        result["unit"],
                        score=result["score"],
                        snippet=result.get("snippet"),
                        include_content=False,
                    ),
                    "reason": result["reason"],
                    "source_mode": result["source_mode"],
                }
                for result in payload.get("results", [])
            ],
            "error": payload.get("error"),
        }.items()
        if value is not None
    }


def _do_similar(
    store: Store,
    unit_id: str,
    *,
    limit: int = 10,
    filters: dict | None = None,
) -> dict:
    from graph.rag.search import RAGService

    filters = filters or {}
    rag = RAGService(store, provider=None)
    payload = rag.similar_units(
        unit_id,
        limit=limit,
        source_project=filters.get("source_project"),
        content_type=filters.get("content_type"),
        tag=filters.get("tag"),
    )
    return _similar_payload_to_json(payload)


def _render_search_payload(payload: dict, *, json_output: bool) -> None:
    if "error" in payload:
        if json_output:
            _json_echo(payload)
        else:
            typer.echo(payload["error"])
        return

    results = payload["results"]
    if json_output:
        _json_echo(payload)
        return

    if not results:
        typer.echo("No results found.")
        return

    for result in results:
        if "score" in result:
            typer.echo(
                f"\n[{result['source_project']}] {result['title']}  (score: {result['score']:.3f})"
            )
        else:
            typer.echo(f"\n[{result['source_project']}] {result['title']}")
        typer.echo(f"  ID: {result['id']}")
        typer.echo(f"  Type: {result['content_type']} | Tags: {', '.join(result['tags'])}")
        typer.echo(f"  {result.get('snippet') or result.get('content', '')[:120]}...")


def _render_similar_payload(payload: dict, *, json_output: bool) -> None:
    if json_output:
        _json_echo(payload)
        return

    if payload.get("error") == "unit_not_found":
        typer.echo(f"Unit not found: {payload['seed_id']}")
        return

    seed = payload.get("seed") or {}
    typer.echo(f"Seed: [{seed.get('source_project')}] {seed.get('title')}")
    typer.echo(f"Mode: {payload.get('source_mode')}")

    results = payload.get("results", [])
    if not results:
        typer.echo("No similar units found.")
        return

    for result in results:
        typer.echo(
            f"\n[{result['source_project']}] {result['title']}  "
            f"(score: {result['score']:.3f}, reason: {result['reason']})"
        )
        typer.echo(f"  ID: {result['id']}")
        typer.echo(f"  Type: {result['content_type']} | Tags: {', '.join(result['tags'])}")
        typer.echo(f"  {result.get('snippet') or ''}")


def _render_search_facets_payload(payload: dict, *, json_output: bool) -> None:
    if "error" in payload:
        if json_output:
            _json_echo(payload)
        else:
            typer.echo(payload["error"])
        return

    if json_output:
        _json_echo(payload)
        return

    typer.echo(
        f"Search facets: {payload['query']} ({payload['mode']}) - "
        f"{payload['total_matches']} matches"
    )
    for facet_name, counts in payload["facets"].items():
        typer.echo(f"\n{facet_name}:")
        if not counts:
            typer.echo("  (none)")
            continue
        for value, count in counts.items():
            typer.echo(f"  {value}: {count}")


def _do_ingest(
    store: Store,
    project: str = "all",
    entity_type: str | None = None,
    full: bool = False,
) -> dict:
    """Core ingest logic. Returns total stats dict."""
    projects = (
        [
            "forty_two",
            "max",
            "presence",
            "me",
            "markdown",
            "kindle",
            "sota",
            "feed",
            "bookmarks",
            "csv",
            "jsonl",
            "opml",
            "text",
            "html",
        ]
        if project == "all"
        else [project]
    )
    entity_types = [entity_type] if entity_type else None

    total_stats = {"units_inserted": 0, "units_skipped": 0, "edges_inserted": 0}

    for proj in projects:
        adapter = _get_adapter_for_project(proj)
        since = None
        if not full:
            for et in adapter.entity_types:
                s = store.get_sync_state(proj, et)
                if s and (since is None or s.last_sync_at < since.last_sync_at):
                    since = s

        mode = "full backfill" if full else "incremental"
        typer.echo(f"Ingesting from {proj} ({mode})...")
        result = adapter.ingest(since=since, entity_types=entity_types)
        stats = store.ingest(result, proj)

        for key in total_stats:
            total_stats[key] += stats[key]

        typer.echo(
            f"  {proj}: {stats['units_inserted']} new, "
            f"{stats['units_skipped']} updated, "
            f"{stats['edges_inserted']} edges"
        )

        for et in adapter.entity_types:
            if entity_types and et not in entity_types:
                continue
            store.upsert_sync_state(
                SyncState(
                    source_project=proj,
                    source_entity_type=et,
                    last_sync_at=datetime.now(timezone.utc),
                    items_synced=stats["units_inserted"],
                )
            )

    typer.echo(
        f"\nTotal: {total_stats['units_inserted']} new units, "
        f"{total_stats['units_skipped']} updated, "
        f"{total_stats['edges_inserted']} edges"
    )
    return total_stats


@app.command()
def ingest(
    project: str = typer.Argument("all", help="Source project or 'all'"),
    entity_type: str | None = typer.Option(None, "--type", "-t", help="Specific entity type"),
    full: bool = typer.Option(
        False,
        "--full",
        help="Ignore sync state and re-upsert all matching source items",
    ),
) -> None:
    """Ingest knowledge from source projects."""
    store = _get_store()
    _do_ingest(store, project=project, entity_type=entity_type, full=full)
    store.close()


@app.command(name="export-json")
def export_json(
    path: Path = typer.Argument(..., help="Destination JSON backup path"),
) -> None:
    """Export the graph to a portable JSON backup."""
    store = _get_store()
    stats = _do_export_json(store, path)
    store.close()
    typer.echo(
        f"Exported {stats['units_exported']} units and "
        f"{stats['edges_exported']} edges to {stats['path']}"
    )


@app.command(name="export-graphml")
def export_graphml(
    path: Path = typer.Argument(..., help="Destination GraphML file path"),
) -> None:
    """Export the graph to GraphML for external visualization tools."""
    store = _get_store()
    stats = _do_export_graphml(store, path)
    store.close()
    typer.echo(
        f"Exported {stats['node_count']} nodes and {stats['edge_count']} edges to {stats['path']}"
    )


@app.command(name="export-mermaid")
def export_mermaid(
    path: Path = typer.Argument(..., help="Destination Markdown file path"),
    unit_id: str | None = typer.Option(
        None,
        "--unit-id",
        help="Center unit ID for a focused neighborhood export",
    ),
    depth: int = typer.Option(1, "--depth", "-d", help="Traversal depth (1-3; max 3)"),
    limit: int = typer.Option(100, "--limit", "-n", help="Maximum nodes to export"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Export a Mermaid Markdown graph for lightweight visualization."""
    store = _get_store()
    try:
        stats = _do_export_mermaid(
            store,
            path,
            unit_id=unit_id,
            depth=depth,
            limit=limit,
        )
    except ValueError as exc:
        store.close()
        try:
            payload = json.loads(str(exc))
        except json.JSONDecodeError:
            payload = {"error": "export_failed", "message": str(exc)}
        if json_output:
            _json_echo(payload)
        else:
            typer.echo(payload["message"])
        raise typer.Exit(code=1) from exc
    store.close()

    if json_output:
        _json_echo(stats)
        return

    capped = " (capped)" if stats["capped"] else ""
    typer.echo(
        f"Exported {stats['node_count']} nodes and "
        f"{stats['edge_count']} edges to {stats['path']}{capped}"
    )


@app.command(name="export-turtle")
def export_turtle(
    path: Path = typer.Argument(..., help="Destination Turtle file path"),
    base_uri: str = typer.Option(
        "https://graph.local/unit/",
        "--base-uri",
        help="Base URI for exported knowledge unit IRIs",
    ),
) -> None:
    """Export the graph to RDF Turtle for linked-data tools."""
    store = _get_store()
    stats = _do_export_turtle(store, path, base_uri=base_uri)
    store.close()
    typer.echo(
        f"Exported {stats['node_count']} nodes and {stats['edge_count']} edges to {stats['path']}"
    )


@app.command(name="export-neighborhood")
def export_neighborhood(
    unit_id: str = typer.Argument(..., help="Center unit ID"),
    path: Path = typer.Argument(..., help="Destination neighborhood JSON path"),
    depth: int = typer.Option(1, "--depth", "-d", help="Traversal depth (1-3; max 3)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Export a focused JSON subgraph around one unit."""
    store = _get_store()
    try:
        stats = _do_export_neighborhood(store, unit_id, path, depth=depth)
    except ValueError as exc:
        store.close()
        try:
            payload = json.loads(str(exc))
        except json.JSONDecodeError:
            payload = {"error": "export_failed", "message": str(exc)}
        if json_output:
            _json_echo(payload)
        else:
            typer.echo(payload["message"])
        raise typer.Exit(code=1) from exc
    store.close()

    if json_output:
        _json_echo(stats)
        return

    typer.echo(
        f"Exported {stats['unit_count']} units and "
        f"{stats['edge_count']} edges to {stats['path']} "
        f"(depth {stats['depth']})"
    )


@app.command(name="export-report")
def export_report(
    path: Path = typer.Argument(..., help="Destination Markdown report path"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max items per section"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Export a Markdown graph health report."""
    store = _get_store()
    stats = _do_export_report(store, path, limit=limit)
    store.close()

    if json_output:
        _json_echo(stats)
        return

    typer.echo(f"Exported graph report to {stats['path']}")


@app.command(name="export-anki")
def export_anki(
    path: Path = typer.Argument(..., help="Destination Anki TSV file path"),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    tag: str | None = typer.Option(None, "--tag", help="Require an exact graph tag"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Maximum rows to export"),
    include_tags: bool = typer.Option(False, "--include-tags", help="Emit graph tags"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Export matching units as Anki-compatible TSV rows."""
    store = _get_store()
    stats = _do_export_anki(
        store,
        path,
        source_project=source_project,
        content_type=content_type,
        tag=tag,
        limit=limit,
        include_tags=include_tags,
    )
    store.close()

    if json_output:
        _json_echo(stats)
        return

    typer.echo(f"Exported {stats['rows_exported']} Anki rows to {stats['path']}")


@app.command(name="import-json")
def import_json_command(
    path: Path = typer.Argument(..., help="Source JSON backup path"),
) -> None:
    """Import a portable JSON graph backup."""
    store = _get_store()
    stats = _do_import_json(store, path)
    store.close()
    typer.echo(
        f"Imported {stats['units_inserted']} units, updated "
        f"{stats['units_updated']} units, inserted {stats['edges_inserted']} edges, "
        f"skipped {stats['edges_skipped']} duplicate edges from {stats['path']}"
    )


def _do_embed(
    store: Store,
    project: str | None = None,
    content_type: str | None = None,
    batch_size: int = 5,
    delay: float = 21.0,
    force: bool = False,
    limit: int | None = None,
    stale_only: bool = False,
) -> int:
    """Core embed logic. Returns number of units embedded."""
    import time

    from graph.rag.embeddings import get_embedding_provider
    from graph.rag.search import RAGService

    provider = get_embedding_provider(
        settings.embedding_provider,
        settings.embedding_api_key,
        settings.embedding_model,
    )
    rag = RAGService(store, provider)

    units_to_embed = [
        unit.id
        for unit in store.get_units_for_embedding_refresh(
            source_project=project,
            content_type=content_type,
            force=force,
            stale_only=stale_only,
            limit=limit,
        )
    ]

    if not units_to_embed:
        if force:
            typer.echo("No units matched the embedding filters.")
        elif stale_only:
            typer.echo("All matching units have fresh embeddings.")
        else:
            typer.echo("All matching units already have embeddings.")
        return 0

    total_batches = (len(units_to_embed) + batch_size - 1) // batch_size
    typer.echo(f"Embedding {len(units_to_embed)} units in {total_batches} batches...")
    total = 0
    for i in range(0, len(units_to_embed), batch_size):
        batch = units_to_embed[i : i + batch_size]
        count = rag.embed_batch_and_store(batch)
        total += count
        batch_num = i // batch_size + 1
        typer.echo(
            f"  Batch {batch_num}/{total_batches}: {count} embedded ({total}/{len(units_to_embed)})"
        )
        if i + batch_size < len(units_to_embed):
            time.sleep(delay)

    typer.echo(f"Done. {total} units embedded.")
    return total


def _do_embeddings_status(
    store: Store,
    *,
    source_project: str | None = None,
    content_type: str | None = None,
    show_stale: int | None = None,
) -> dict:
    stale_limit = max(0, show_stale) if show_stale is not None else 0
    return {
        "filters": {
            "source_project": source_project,
            "content_type": content_type,
        },
        "totals": store.get_embedding_status(
            source_project=source_project,
            content_type=content_type,
        ),
        "by_source_project": store.get_embedding_status_groups(
            "source_project",
            source_project=source_project,
            content_type=content_type,
        ),
        "by_content_type": store.get_embedding_status_groups(
            "content_type",
            source_project=source_project,
            content_type=content_type,
        ),
        "groups": store.get_embedding_status_matrix(
            source_project=source_project,
            content_type=content_type,
        ),
        "stale_units": store.get_embedding_refresh_status(
            source_project=source_project,
            content_type=content_type,
            limit=stale_limit,
        )
        if stale_limit
        else [],
        "stale_limit": stale_limit,
    }


def _legacy_embeddings_status(
    store: Store,
    *,
    project: str | None = None,
    content_type: str | None = None,
) -> dict:
    status = store.get_embedding_status(
        source_project=project,
        content_type=content_type,
    )
    if project is not None:
        status["source_project"] = project
    if content_type is not None:
        status["content_type"] = content_type
    return status


def _do_infer_edges(
    store: Store,
    *,
    source_project: str | None = None,
    content_type: str | None = None,
    min_similarity: float = 0.75,
    limit: int = 100,
    dry_run: bool = False,
) -> dict:
    from graph.rag.search import RAGService

    rag = RAGService(store, provider=None)
    return rag.infer_similarity_edges(
        min_similarity=min_similarity,
        limit=limit,
        source_project=source_project,
        content_type=content_type,
        dry_run=dry_run,
    )


def _do_integrity_audit(
    store: Store,
    *,
    repair_fts: bool = False,
    limit: int = 20,
) -> dict:
    from graph.graph.service import GraphService

    gs = GraphService(store)
    return gs.integrity_audit(repair_fts=repair_fts, limit=limit)


@app.command()
def embed(
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by source project"),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    batch_size: int = typer.Option(5, "--batch-size", "-b", help="Batch size for API calls"),
    delay: float = typer.Option(21.0, "--delay", help="Seconds between batches (rate limit)"),
    force: bool = typer.Option(
        False, "--force", help="Refresh embeddings even when they are fresh"
    ),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Maximum units to embed"),
    stale_only: bool = typer.Option(
        False, "--stale-only", help="Embed only missing or stale embeddings"
    ),
) -> None:
    """Generate embeddings for units missing them."""
    store = _get_store()
    _do_embed(
        store,
        project=project,
        content_type=content_type,
        batch_size=batch_size,
        delay=delay,
        force=force,
        limit=limit,
        stale_only=stale_only,
    )
    store.close()


@app.command(name="embeddings-status")
def embeddings_status(
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by source project"),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Report embedding freshness counts."""
    store = _get_store()
    status = _legacy_embeddings_status(
        store,
        project=project,
        content_type=content_type,
    )
    store.close()

    if json_output:
        _json_echo(status)
        return

    filters = []
    if project is not None:
        filters.append(f"project={project}")
    if content_type is not None:
        filters.append(f"content_type={content_type}")
    suffix = f" ({', '.join(filters)})" if filters else ""
    typer.echo(f"Embedding status{suffix}:")
    typer.echo(f"  Total: {status['total']}")
    typer.echo(f"  Fresh: {status['fresh']}")
    typer.echo(f"  Stale: {status['stale']}")
    typer.echo(f"  Missing: {status['missing']}")
    typer.echo(f"  Percent fresh: {status['percent_fresh']}%")


@app.command(name="embedding-status")
def embedding_status(
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    show_stale: int | None = typer.Option(
        None,
        "--show-stale",
        min=0,
        help="List up to LIMIT missing or stale units needing refresh",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Report embedding coverage and stale units by source and content type."""
    store = _get_store()
    report = _do_embeddings_status(
        store,
        source_project=source_project,
        content_type=content_type,
        show_stale=show_stale,
    )
    store.close()

    if json_output:
        _json_echo(report)
        return

    filters = []
    if source_project is not None:
        filters.append(f"source_project={source_project}")
    if content_type is not None:
        filters.append(f"content_type={content_type}")
    suffix = f" ({', '.join(filters)})" if filters else ""
    totals = report["totals"]
    typer.echo(f"Embedding coverage{suffix}:")
    typer.echo(
        f"  Total: {totals['total']} | Fresh: {totals['fresh']} "
        f"({totals['percent_fresh']}%) | Stale: {totals['stale']} | Missing: {totals['missing']}"
    )
    typer.echo("\nBy source project:")
    for group in report["by_source_project"]:
        typer.echo(
            f"  {group['source_project']}: {group['fresh']}/{group['total']} fresh "
            f"({group['percent_fresh']}%), {group['stale']} stale, {group['missing']} missing"
        )
    typer.echo("\nBy content type:")
    for group in report["by_content_type"]:
        typer.echo(
            f"  {group['content_type']}: {group['fresh']}/{group['total']} fresh "
            f"({group['percent_fresh']}%), {group['stale']} stale, {group['missing']} missing"
        )
    if show_stale is not None:
        typer.echo(f"\nUnits needing refresh (limit {report['stale_limit']}):")
        if not report["stale_units"]:
            typer.echo("  None")
        for unit in report["stale_units"]:
            typer.echo(
                f"  {unit['id']} | [{unit['source_project']}/{unit['content_type']}] "
                f"{unit['title']} - {unit['reason']}"
            )


@app.command(name="infer-edges")
def infer_edges(
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter candidate units by source project",
    ),
    content_type: str | None = typer.Option(
        None, "--content-type", help="Filter candidate units by content type"
    ),
    min_similarity: float = typer.Option(
        0.75, "--min-similarity", help="Minimum cosine similarity"
    ),
    limit: int = typer.Option(100, "--limit", "-n", help="Max candidate pairs to process"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview inferred edges without writing"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Infer RELATES_TO edges between semantically similar embedded units."""
    store = _get_store()
    result = _do_infer_edges(
        store,
        source_project=source_project,
        content_type=content_type,
        min_similarity=min_similarity,
        limit=limit,
        dry_run=dry_run,
    )

    if json_output:
        _json_echo(result)
        store.close()
        return

    action = "Would insert" if dry_run else "Inserted"
    typer.echo(
        f"{action} {result['inserted'] if not dry_run else len([c for c in result['candidates'] if c['status'] == 'would_insert'])} "
        f"inferred edges; skipped {result['skipped']} existing direct edges."
    )
    for candidate in result["candidates"]:
        typer.echo(
            f"  {candidate['status']}: {candidate['from_title']} -> "
            f"{candidate['to_title']} ({candidate['similarity']:.3f})"
        )

    store.close()


@app.command(name="update-unit")
def update_unit(
    unit_id: str = typer.Argument(..., help="Knowledge unit ID"),
    title: str | None = typer.Option(None, "--title", help="Replace the unit title"),
    content: str | None = typer.Option(None, "--content", help="Replace the unit content"),
    content_type: str | None = typer.Option(
        None,
        "--content-type",
        help="Replace the content type",
    ),
    tag: list[str] = typer.Option(
        [],
        "--tag",
        help="Append a tag. Repeat to add multiple tags.",
    ),
    metadata_json: str | None = typer.Option(
        None,
        "--metadata-json",
        help="Merge a JSON object into existing metadata",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Update a manually maintained knowledge unit."""
    store = _get_store()
    try:
        payload = _do_update_unit(
            store,
            unit_id,
            title=title,
            content=content,
            content_type=content_type,
            tags=tag,
            metadata=metadata_json,
        )
    except ValueError as exc:
        if json_output:
            _json_echo({"unit_id": unit_id, "updated": False, "error": str(exc)})
            return
        raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()

    if payload.get("error"):
        if json_output:
            _json_echo(payload)
            return
        typer.echo(payload["message"])
        raise typer.Exit(code=1)

    if json_output:
        _json_echo(payload)
        return

    unit = payload["unit"]
    typer.echo(f"Updated unit {unit['id']}: {unit['title']}")


@app.command(name="pin-unit")
def pin_unit(
    unit_id: str = typer.Argument(..., help="Knowledge unit ID"),
    reason: str | None = typer.Option(None, "--reason", help="Optional pin reason"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Pin a knowledge unit for later retrieval."""
    store = _get_store()
    payload = _do_pin_unit(store, unit_id, reason=reason)
    store.close()

    if payload.get("error"):
        if json_output:
            _json_echo(payload)
            return
        typer.echo(payload["message"])
        raise typer.Exit(code=1)

    if json_output:
        _json_echo(payload)
        return

    unit = payload["unit"]
    typer.echo(f"Pinned unit {unit['id']}: {unit['title']}")


@app.command(name="unpin-unit")
def unpin_unit(
    unit_id: str = typer.Argument(..., help="Knowledge unit ID"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Remove pin metadata from a knowledge unit."""
    store = _get_store()
    payload = _do_unpin_unit(store, unit_id)
    store.close()

    if payload.get("error"):
        if json_output:
            _json_echo(payload)
            return
        typer.echo(payload["message"])
        raise typer.Exit(code=1)

    if json_output:
        _json_echo(payload)
        return

    unit = payload["unit"]
    typer.echo(f"Unpinned unit {unit['id']}: {unit['title']}")


@app.command(name="pinned")
def pinned(
    source_project: str | None = typer.Option(
        None, "--source-project", help="Filter by source project"
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    tag: str | None = typer.Option(None, "--tag", help="Require an exact graph tag"),
    limit: int = typer.Option(20, "--limit", min=0, help="Maximum pinned units to return"),
    include_content: bool = typer.Option(
        False,
        "--include-content",
        help="Include full content in JSON output",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """List pinned knowledge units newest pin first."""
    store = _get_store()
    try:
        payload = _do_pinned_units(
            store,
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            limit=limit,
            include_content=include_content,
        )
    finally:
        store.close()

    if json_output:
        _json_echo(payload)
        return

    if not payload["units"]:
        typer.echo("No pinned units found.")
        return

    for unit in payload["units"]:
        reason = f" - {unit['pin_reason']}" if unit.get("pin_reason") else ""
        typer.echo(
            f"{unit['pinned_at']} {unit['id']} [{unit['source_project']}] {unit['title']}{reason}"
        )


@app.command(name="delete-unit")
def delete_unit(
    unit_id: str = typer.Argument(..., help="Knowledge unit ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without prompting"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Delete a knowledge unit, its FTS row, and related edges."""
    if not yes:
        typer.confirm(f"Delete unit {unit_id} and its related edges?", abort=True)

    store = _get_store()
    payload = _do_delete_unit(store, unit_id)
    store.close()

    if payload.get("error"):
        if json_output:
            _json_echo(payload)
            return
        typer.echo(payload["message"])
        raise typer.Exit(code=1)

    if json_output:
        _json_echo(payload)
        return

    typer.echo(f"Deleted unit {unit_id}; removed {payload['edges_deleted']} related edges.")


@units_app.command(name="merge")
def merge_units(
    source_id: str = typer.Argument(..., help="Source knowledge unit ID to merge and delete"),
    target_id: str = typer.Argument(..., help="Target knowledge unit ID to keep"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview merge effects without writing"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Merge a duplicate source unit into a target unit."""
    store = _get_store()
    try:
        payload = _do_merge_units(store, source_id, target_id, dry_run=dry_run)
    except ValueError as exc:
        payload = {
            "source_id": source_id,
            "target_id": target_id,
            "dry_run": dry_run,
            "merged": False,
            "error": str(exc),
        }
    finally:
        store.close()

    if payload.get("error"):
        if json_output:
            _json_echo(payload)
            return
        typer.echo(payload.get("message", payload["error"]))
        raise typer.Exit(code=1)

    if json_output:
        _json_echo(payload)
        return

    action = "Would merge" if dry_run else "Merged"
    typer.echo(
        f"{action} {source_id} into {target_id}: "
        f"{payload['rewired_edge_counts']['total']} edges rewired, "
        f"{len(payload['skipped_duplicate_edges'])} duplicate edges skipped, "
        f"{len(payload['skipped_self_edges'])} self edges skipped."
    )


@edges_app.command(name="list")
def list_edges(
    unit_id: str = typer.Argument(..., help="Knowledge unit ID"),
    direction: str = typer.Option(
        "both",
        "--direction",
        help="Edge direction: incoming | outgoing | both",
    ),
    relation: str | None = typer.Option(None, "--relation", help="Filter by edge relation"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """List a unit's incoming and outgoing edges."""
    store = _get_store()
    try:
        payload = _list_edges_payload(
            store,
            unit_id,
            direction=direction,
            relation=relation,
        )
    except ValueError as exc:
        if json_output:
            _json_echo({"unit_id": unit_id, "edges": [], "error": str(exc)})
            return
        raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()

    if payload.get("error"):
        if json_output:
            _json_echo(payload)
            return
        typer.echo(payload["message"])
        raise typer.Exit(code=1)

    if json_output:
        _json_echo(payload)
        return

    center = payload["center"]
    typer.echo(f"Center: [{center['source_project']}] {center['title']}")
    if not payload["edges"]:
        typer.echo("No edges found.")
        return
    for edge in payload["edges"]:
        neighbor = edge.get("neighbor") or {}
        typer.echo(
            f"  {edge['direction']}: {edge['id']} "
            f"{edge['from_unit_id'][:8]}... --{edge['relation']}--> "
            f"{edge['to_unit_id'][:8]}... "
            f"(weight: {edge['weight']}, source: {edge['source']})"
        )
        if neighbor:
            typer.echo(
                f"    Neighbor: [{neighbor['source_project']}] "
                f"{neighbor['title']} ({neighbor['id'][:8]}...)"
            )


@edges_app.command(name="import-csv")
def import_edges_csv(
    path: Path = typer.Argument(..., help="CSV file with graph edges to import"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without inserting edges"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Import graph edges from a CSV file."""
    store = _get_store()
    try:
        payload = _do_import_edges_csv(store, path, dry_run=dry_run)
    except OSError as exc:
        if json_output:
            _json_echo(
                {
                    "path": str(path),
                    "dry_run": dry_run,
                    "inserted": 0,
                    "skipped_existing": 0,
                    "invalid": [{"row_number": None, "reasons": [str(exc)]}],
                    "inserted_rows": [],
                    "skipped_existing_rows": [],
                    "error": "file_error",
                }
            )
            return
        raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()

    if json_output:
        _json_echo(payload)
        return

    action = "Would insert" if dry_run else "Inserted"
    typer.echo(
        f"{action} {payload['inserted']} edges from {payload['path']}; "
        f"skipped {payload['skipped_existing']} existing; "
        f"invalid {len(payload['invalid'])} rows."
    )
    for invalid in payload["invalid"]:
        typer.echo(
            f"  Row {invalid['row_number']}: "
            f"{'; '.join(invalid['reasons'])}"
        )


@app.command(name="update-edge")
def update_edge(
    edge_id: str = typer.Argument(..., help="Edge ID"),
    relation: str | None = typer.Option(None, "--relation", help="Replace the edge relation"),
    weight: float | None = typer.Option(None, "--weight", help="Replace the edge weight"),
    source: str | None = typer.Option(None, "--source", help="Replace the edge source"),
    metadata_json: str | None = typer.Option(
        None,
        "--metadata-json",
        help="Merge a JSON object into existing edge metadata",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Update a graph edge by ID."""
    store = _get_store()
    try:
        payload = _do_update_edge(
            store,
            edge_id,
            relation=relation,
            weight=weight,
            source=source,
            metadata=metadata_json,
        )
    except ValueError as exc:
        if json_output:
            _json_echo({"edge_id": edge_id, "updated": False, "error": str(exc)})
            return
        raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()

    if payload.get("error"):
        if json_output:
            _json_echo(payload)
            return
        typer.echo(payload["message"])
        raise typer.Exit(code=1)

    if json_output:
        _json_echo(payload)
        return

    edge = payload["edge"]
    typer.echo(f"Updated edge {edge['id']}: {edge['relation']}")


@app.command(name="delete-edge")
def delete_edge(
    edge_id: str = typer.Argument(..., help="Edge ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without prompting"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Delete a graph edge by ID."""
    if not yes:
        typer.confirm(f"Delete edge {edge_id}?", abort=True)

    store = _get_store()
    payload = _do_delete_edge(store, edge_id)
    store.close()

    if payload.get("error"):
        if json_output:
            _json_echo(payload)
            return
        typer.echo(payload["message"])
        raise typer.Exit(code=1)

    if json_output:
        _json_echo(payload)
        return

    typer.echo(f"Deleted edge {edge_id}.")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    mode: str = typer.Option(
        "fulltext", "--mode", "-m", help="Search mode: fulltext | semantic | hybrid"
    ),
    sort: str = typer.Option(
        "relevance",
        "--sort",
        help="Sort order: relevance | created_at_desc | created_at_asc | updated_at_desc | utility_desc | confidence_desc",
    ),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    tag: str | None = typer.Option(None, "--tag", help="Require an exact tag"),
    review_state: str | None = typer.Option(
        None,
        "--review-state",
        help="Filter by review state metadata",
    ),
    created_after: str | None = typer.Option(
        None,
        "--created-after",
        help="Filter to units created on or after an ISO-8601 date/datetime",
    ),
    created_before: str | None = typer.Option(
        None,
        "--created-before",
        help="Filter to units created on or before an ISO-8601 date/datetime",
    ),
    updated_after: str | None = typer.Option(
        None,
        "--updated-after",
        help="Filter to units updated on or after an ISO-8601 date/datetime",
    ),
    updated_before: str | None = typer.Option(
        None,
        "--updated-before",
        help="Filter to units updated on or before an ISO-8601 date/datetime",
    ),
    min_utility: float | None = typer.Option(
        None,
        "--min-utility",
        help="Filter to units with utility score at least this value",
    ),
    max_utility: float | None = typer.Option(
        None,
        "--max-utility",
        help="Filter to units with utility score at most this value",
    ),
    min_confidence: float | None = typer.Option(
        None,
        "--min-confidence",
        help="Filter to units with confidence at least this value",
    ),
    max_confidence: float | None = typer.Option(
        None,
        "--max-confidence",
        help="Filter to units with confidence at most this value",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Search knowledge units."""
    store = _get_store()
    try:
        payload = _do_search(
            store,
            query,
            limit=limit,
            mode=mode,
            sort=sort,
            filters=_search_filters_dict(
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                review_state=review_state,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                min_utility=min_utility,
                max_utility=max_utility,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
            ),
        )
    except ValueError as exc:
        payload = {
            "error": str(exc),
            "valid_modes": ["fulltext", "semantic", "hybrid"],
            "valid_sorts": list(SEARCH_SORTS),
        }

    _render_search_payload(payload, json_output=json_output)
    store.close()


@app.command()
def similar(
    unit_id: str = typer.Argument(..., help="Seed unit id"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    tag: str | None = typer.Option(None, "--tag", help="Require an exact tag"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Find units similar to an existing unit."""
    store = _get_store()
    payload = _do_similar(
        store,
        unit_id,
        limit=limit,
        filters=_search_filters_dict(
            source_project=source_project,
            content_type=content_type,
            tag=tag,
        ),
    )
    _render_similar_payload(payload, json_output=json_output)
    store.close()


@app.command(name="search-facets")
def search_facets(
    query: str = typer.Argument(..., help="Search query"),
    mode: str = typer.Option(
        "fulltext", "--mode", "-m", help="Search mode: fulltext | semantic | hybrid"
    ),
    sort: str = typer.Option(
        "relevance",
        "--sort",
        help="Accepted for parity with search; facets are count summaries.",
    ),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    tag: str | None = typer.Option(None, "--tag", help="Require an exact tag"),
    review_state: str | None = typer.Option(
        None,
        "--review-state",
        help="Filter by review state metadata",
    ),
    created_after: str | None = typer.Option(
        None,
        "--created-after",
        help="Filter to units created on or after an ISO-8601 date/datetime",
    ),
    created_before: str | None = typer.Option(
        None,
        "--created-before",
        help="Filter to units created on or before an ISO-8601 date/datetime",
    ),
    updated_after: str | None = typer.Option(
        None,
        "--updated-after",
        help="Filter to units updated on or after an ISO-8601 date/datetime",
    ),
    updated_before: str | None = typer.Option(
        None,
        "--updated-before",
        help="Filter to units updated on or before an ISO-8601 date/datetime",
    ),
    min_utility: float | None = typer.Option(
        None,
        "--min-utility",
        help="Filter to units with utility score at least this value",
    ),
    max_utility: float | None = typer.Option(
        None,
        "--max-utility",
        help="Filter to units with utility score at most this value",
    ),
    min_confidence: float | None = typer.Option(
        None,
        "--min-confidence",
        help="Filter to units with confidence at least this value",
    ),
    max_confidence: float | None = typer.Option(
        None,
        "--max-confidence",
        help="Filter to units with confidence at most this value",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Summarize matching knowledge units by faceted counts."""
    store = _get_store()
    try:
        payload = _do_search_facets(
            store,
            query,
            mode=mode,
            sort=sort,
            filters=_search_filters_dict(
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                review_state=review_state,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                min_utility=min_utility,
                max_utility=max_utility,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
            ),
        )
    except ValueError as exc:
        payload = {
            "error": str(exc),
            "valid_modes": ["fulltext", "semantic", "hybrid"],
            "valid_sorts": list(SEARCH_SORTS),
        }

    _render_search_facets_payload(payload, json_output=json_output)
    store.close()


@app.command(name="context")
def context(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max ranked units"),
    mode: str = typer.Option(
        "fulltext", "--mode", "-m", help="Search mode: fulltext | semantic | hybrid"
    ),
    sort: str = typer.Option(
        "relevance",
        "--sort",
        help="Sort order for ranked units",
    ),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    tag: str | None = typer.Option(None, "--tag", help="Require an exact tag"),
    review_state: str | None = typer.Option(
        None,
        "--review-state",
        help="Filter by review state metadata",
    ),
    created_after: str | None = typer.Option(
        None,
        "--created-after",
        help="Filter to units created on or after an ISO-8601 date/datetime",
    ),
    created_before: str | None = typer.Option(
        None,
        "--created-before",
        help="Filter to units created on or before an ISO-8601 date/datetime",
    ),
    updated_after: str | None = typer.Option(
        None,
        "--updated-after",
        help="Filter to units updated on or after an ISO-8601 date/datetime",
    ),
    updated_before: str | None = typer.Option(
        None,
        "--updated-before",
        help="Filter to units updated on or before an ISO-8601 date/datetime",
    ),
    min_utility: float | None = typer.Option(
        None,
        "--min-utility",
        help="Filter to units with utility score at least this value",
    ),
    max_utility: float | None = typer.Option(
        None,
        "--max-utility",
        help="Filter to units with utility score at most this value",
    ),
    neighbor_depth: int = typer.Option(
        1,
        "--neighbor-depth",
        help="Graph neighbor depth for context; capped at 2",
    ),
    char_budget: int = typer.Option(
        4000,
        "--char-budget",
        help="Total character budget for content excerpts",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Build a compact retrieval context pack for LLM use."""
    store = _get_store()
    try:
        payload = _do_context_pack(
            store,
            query,
            limit=limit,
            mode=mode,
            sort=sort,
            filters=_search_filters_dict(
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                review_state=review_state,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                min_utility=min_utility,
                max_utility=max_utility,
            ),
            char_budget=char_budget,
            neighbor_depth=neighbor_depth,
        )
    except ValueError as exc:
        payload = {
            "error": str(exc),
            "valid_modes": ["fulltext", "semantic", "hybrid"],
            "valid_sorts": list(SEARCH_SORTS),
        }
    finally:
        store.close()

    if json_output:
        _json_echo(payload)
        return
    if payload.get("error"):
        typer.echo(payload["error"])
        raise typer.Exit(code=1)
    if not payload["ranked_units"]:
        typer.echo("No results found.")
        return
    typer.echo(f"Context pack: {payload['query']} ({payload['mode']})")
    for unit in payload["ranked_units"]:
        score = f" | score {unit['score']:.3f}" if "score" in unit else ""
        typer.echo(f"{unit['rank']}. [{unit['source_project']}] {unit['title']}{score}")
        typer.echo(f"   neighbors: {len(unit['neighbor_ids'])}")


@queries_app.command(name="save")
def save_query(
    name: str = typer.Argument(..., help="Saved query name"),
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    mode: str = typer.Option(
        "fulltext", "--mode", "-m", help="Search mode: fulltext | semantic | hybrid"
    ),
    sort: str = typer.Option(
        "relevance",
        "--sort",
        help="Sort order saved with the query filters",
    ),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    tag: str | None = typer.Option(None, "--tag", help="Require an exact tag"),
    review_state: str | None = typer.Option(
        None,
        "--review-state",
        help="Filter by review state metadata",
    ),
    created_after: str | None = typer.Option(
        None,
        "--created-after",
        help="Filter to units created on or after an ISO-8601 date/datetime",
    ),
    created_before: str | None = typer.Option(
        None,
        "--created-before",
        help="Filter to units created on or before an ISO-8601 date/datetime",
    ),
    updated_after: str | None = typer.Option(
        None,
        "--updated-after",
        help="Filter to units updated on or after an ISO-8601 date/datetime",
    ),
    updated_before: str | None = typer.Option(
        None,
        "--updated-before",
        help="Filter to units updated on or before an ISO-8601 date/datetime",
    ),
    min_utility: float | None = typer.Option(
        None,
        "--min-utility",
        help="Filter to units with utility score at least this value",
    ),
    max_utility: float | None = typer.Option(
        None,
        "--max-utility",
        help="Filter to units with utility score at most this value",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Save a reusable search query."""
    try:
        _validate_search_mode(mode)
        validate_search_sort(sort)
        filters = _search_filters_dict(
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            review_state=review_state,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            min_utility=min_utility,
            max_utility=max_utility,
            sort=sort if sort != "relevance" else None,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    store = _get_store()
    saved = store.save_query(
        name=name,
        query=query,
        mode=mode,
        limit=limit,
        filters=filters,
    )
    store.close()

    if json_output:
        _json_echo(saved)
        return
    typer.echo(f"Saved query '{name}' ({mode}, limit {limit}).")


@queries_app.command(name="list")
def list_queries(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """List saved queries."""
    store = _get_store()
    queries = store.list_saved_queries()
    store.close()

    if json_output:
        _json_echo({"queries": queries})
        return

    if not queries:
        typer.echo("No saved queries.")
        return

    for saved in queries:
        filters = ", ".join(f"{key}={value}" for key, value in saved["filters"].items())
        suffix = f" | {filters}" if filters else ""
        typer.echo(
            f"{saved['name']}: {saved['query']} ({saved['mode']}, limit {saved['limit']}){suffix}"
        )


@queries_app.command(name="export")
def export_queries(
    path: Path = typer.Argument(..., help="Destination saved queries JSON path"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Export saved queries to a portable JSON file."""
    store = _get_store()
    stats = _do_export_queries(store, path)
    store.close()

    if json_output:
        _json_echo(stats)
        return

    typer.echo(f"Exported {stats['exported']} saved queries to {stats['path']}")


@queries_app.command(name="import")
def import_queries(
    path: Path = typer.Argument(..., help="Source saved queries JSON path"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Import saved queries from a portable JSON file."""
    store = _get_store()
    try:
        stats = _do_import_queries(store, path)
    except ValueError as exc:
        store.close()
        payload = {"error": "import_failed", "message": str(exc), "path": str(path)}
        if json_output:
            _json_echo(payload)
        else:
            typer.echo(payload["message"])
        raise typer.Exit(code=1) from exc
    store.close()

    if json_output:
        _json_echo(stats)
        return

    typer.echo(
        f"Imported {stats['inserted']} saved queries, updated "
        f"{stats['updated']} saved queries, skipped {stats['skipped']} saved queries "
        f"from {stats['path']}"
    )


@queries_app.command(name="run")
def run_query(
    name: str = typer.Argument(..., help="Saved query name"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Run a saved query."""
    store = _get_store()
    saved = store.get_saved_query(name)
    if not saved:
        store.close()
        if json_output:
            _json_echo({"error": f"Saved query not found: {name}", "name": name})
            return
        typer.echo(f"Saved query not found: {name}")
        raise typer.Exit(code=1)

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
        }

    _render_search_payload(payload, json_output=json_output)
    store.close()


@queries_app.command(name="delete")
def delete_query(
    name: str = typer.Argument(..., help="Saved query name"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Delete a saved query."""
    store = _get_store()
    deleted = store.delete_saved_query(name)
    store.close()

    payload = {"name": name, "deleted": deleted}
    if not deleted:
        payload["error"] = f"Saved query not found: {name}"

    if json_output:
        _json_echo(payload)
        return

    if deleted:
        typer.echo(f"Deleted saved query '{name}'.")
        return
    typer.echo(f"Saved query not found: {name}")
    raise typer.Exit(code=1)


@app.command()
def ideas(
    review_state: str | None = typer.Option(
        None,
        "--review-state",
        help="Filter max ideas by review state, e.g. approved, rejected, unreviewed",
    ),
    approved: bool = typer.Option(
        False,
        "--approved",
        help="Shortcut for --review-state approved",
    ),
    domain: str | None = typer.Option(None, "--domain", help="Filter by idea domain"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """List max ideas stored in the graph with review metadata."""
    store = _get_store()
    wanted_review_state = "approved" if approved else review_state

    found = []
    for unit in store.get_all_units(limit=100000):
        if unit.source_project != "max" or unit.source_entity_type != "buildable_unit":
            continue
        metadata = unit.metadata
        if wanted_review_state and metadata.get("review_state") != wanted_review_state:
            continue
        if domain and metadata.get("domain") != domain:
            continue
        found.append(unit)
        if len(found) >= limit:
            break

    if not found:
        if json_output:
            _json_echo({"results": []})
            store.close()
            return
        typer.echo("No matching ideas found.")
        store.close()
        return

    if json_output:
        _json_echo({"results": [_unit_to_json(unit) for unit in found]})
        store.close()
        return

    for unit in found:
        metadata = unit.metadata
        typer.echo(f"\n[{metadata.get('review_state', 'unknown')}] {unit.title}")
        typer.echo(f"  ID: {unit.id} | Source: {unit.source_id}")
        typer.echo(f"  Domain: {metadata.get('domain') or '-'} | Tags: {', '.join(unit.tags)}")
        buyer = metadata.get("buyer")
        if buyer:
            typer.echo(f"  Buyer: {buyer}")
        reason = metadata.get("feedback_reason")
        if reason:
            typer.echo(f"  Review reason: {reason}")
        typer.echo(f"  {unit.content[:160]}...")

    store.close()


@app.command(name="design-briefs")
def design_briefs(
    status: str | None = typer.Option(None, "--status", help="Filter by design brief status"),
    domain: str | None = typer.Option(None, "--domain", help="Filter by brief domain"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """List max design briefs stored in the graph."""
    store = _get_store()

    found = []
    for unit in store.get_all_units(limit=100000):
        if unit.source_project != "max" or unit.source_entity_type != "design_brief":
            continue

        metadata = unit.metadata
        brief_status = metadata.get("design_status") or metadata.get("status")
        if status and str(brief_status or "").lower() != status.lower():
            continue
        if domain and metadata.get("domain") != domain:
            continue

        found.append(unit)
        if len(found) >= limit:
            break

    if not found:
        if json_output:
            _json_echo({"results": []})
            store.close()
            return
        typer.echo("No matching design briefs found.")
        store.close()
        return

    if json_output:
        _json_echo({"results": [_unit_to_json(unit) for unit in found]})
        store.close()
        return

    for unit in found:
        metadata = unit.metadata
        readiness_score = metadata.get("readiness_score")
        brief_status = metadata.get("design_status") or metadata.get("status")
        lead_idea_id = metadata.get("lead_idea_id")
        source_idea_ids = metadata.get("source_idea_ids") or []
        first_milestones = metadata.get("first_milestones") or []

        typer.echo(f"\n[{brief_status or 'unknown'}] {unit.title}")
        typer.echo(f"  ID: {unit.id} | Source: {unit.source_id}")
        typer.echo(
            f"  Domain: {metadata.get('domain') or '-'} | Theme: {metadata.get('theme') or '-'}"
        )
        typer.echo(f"  Readiness: {readiness_score if readiness_score is not None else '-'}")
        typer.echo(f"  Status: {brief_status or '-'}")
        typer.echo(
            f"  Lead idea: {lead_idea_id or '-'} | Source ideas: "
            f"{', '.join(source_idea_ids) if source_idea_ids else '-'}"
        )

        validation_plan = metadata.get("validation_plan")
        if validation_plan:
            typer.echo(f"  Validation plan: {validation_plan}")
        if first_milestones:
            typer.echo("  First milestones:")
            for milestone in first_milestones:
                typer.echo(f"    - {milestone}")

        typer.echo(f"  {unit.content[:160]}...")

    store.close()


@app.command()
def stats(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Show graph statistics."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()
    s = gs.stats()

    if json_output:
        _json_echo(s)
        store.close()
        return

    typer.echo(f"Nodes: {s['nodes']}")
    typer.echo(f"Edges: {s['edges']}")
    typer.echo(f"Components: {s['components']}")
    typer.echo(f"Density: {s['density']}")
    typer.echo("\nBy project:")
    for proj, count in s["by_project"].items():
        typer.echo(f"  {proj}: {count}")
    typer.echo("\nBy content type:")
    for ct, count in s["by_content_type"].items():
        typer.echo(f"  {ct}: {count}")

    store.close()


@app.command()
def integrity(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
    repair_fts: bool = typer.Option(
        False,
        "--repair-fts",
        help="Restore missing FTS rows and remove stale FTS rows",
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum examples per audit category"),
) -> None:
    """Audit graph table integrity."""
    store = _get_store()
    payload = _do_integrity_audit(store, repair_fts=repair_fts, limit=limit)
    store.close()

    if json_output:
        _json_echo(payload)
        return

    typer.echo(f"Issues: {payload['issue_count']}")
    for name, category in payload["categories"].items():
        typer.echo(f"  {name}: {category['count']}")
    if payload["repair"]["requested"]:
        typer.echo(
            "FTS repair: "
            f"{payload['repair']['fts_rows_inserted']} inserted, "
            f"{payload['repair']['fts_rows_deleted']} deleted"
        )


@app.command()
def serve() -> None:
    """Start the MCP server (stdio transport)."""
    import asyncio

    from graph.mcp.server import run_mcp_server

    asyncio.run(run_mcp_server())


@app.command(name="sync-status")
def sync_status(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Show last sync timestamps per source project."""
    store = _get_store()

    targets = _supported_sync_targets()
    if json_output:
        statuses = []
        for proj, et in targets:
            state = store.get_sync_state(proj, et)
            statuses.append(
                {
                    "source_project": proj,
                    "source_entity_type": et,
                    "last_sync_at": state.last_sync_at if state else None,
                    "last_source_id": state.last_source_id if state else None,
                    "items_synced": state.items_synced if state else 0,
                    "synced": state is not None,
                }
            )
        _json_echo({"results": statuses})
        store.close()
        return

    current_project = None
    for proj, et in targets:
        if proj != current_project:
            typer.echo(f"\n{proj}:")
            current_project = proj
        state = store.get_sync_state(proj, et)
        if state:
            typer.echo(f"  {et}: last sync {state.last_sync_at} ({state.items_synced} total)")
        else:
            typer.echo(f"  {et}: never synced")

    store.close()


@app.command(name="freshness")
def freshness(
    days: int = typer.Option(
        7,
        "--days",
        min=0,
        help="Recent ingest window and stale sync threshold in days",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Report source freshness by sync target."""
    store = _get_store()
    report = _do_freshness_report(store, days=days)
    store.close()

    if json_output:
        _json_echo(report)
        return

    typer.echo(f"Source freshness ({report['days']} day window):")
    for item in report["results"]:
        last_sync = item["last_sync_at"] or "never"
        age = "-" if item["age_days"] is None else f"{item['age_days']:.1f}d"
        stale = "stale" if item["stale"] else "fresh"
        typer.echo(
            f"  {item['source_project']}/{item['source_entity_type']}: "
            f"{stale}, last sync {last_sync}, age {age}, "
            f"recent {item['recent_unit_count']}, total {item['total_unit_count']}"
        )


@app.command()
def backlinks(
    unit_id: str = typer.Argument(..., help="Unit ID"),
    direction: str = typer.Option(
        "both",
        "--direction",
        help="Edge direction: incoming | outgoing | both",
    ),
    relation: str | None = typer.Option(None, "--relation", help="Filter by edge relation"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max links to return"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Show expanded incoming and outgoing references for a unit."""
    if direction not in ("incoming", "outgoing", "both"):
        raise typer.BadParameter("Use incoming, outgoing, or both.", param_hint="--direction")

    store = _get_store()
    try:
        payload = _backlinks_payload(
            store,
            unit_id,
            direction=direction,
            relation=relation,
            limit=limit,
        )
    finally:
        store.close()

    if payload.get("error"):
        if json_output:
            _json_echo(payload)
            return
        typer.echo(payload["message"])
        raise typer.Exit(code=1)

    if json_output:
        _json_echo(payload)
        return

    center = payload["center"]
    typer.echo(f"Center: [{center['source_project']}] {center['title']}")
    if not payload["links"]:
        typer.echo("No backlinks found.")
        return

    for link in payload["links"]:
        unit = link["unit"]
        edge = link["edge"]
        arrow = "<--" if link["direction"] == "incoming" else "-->"
        typer.echo(
            f"  {link['direction']}: {arrow}{link['relation']} "
            f"[{unit['source_project']}] {unit['title']} ({unit['id'][:8]}...)"
        )
        typer.echo(
            f"    Edge: {edge['from_unit_id'][:8]}... -> "
            f"{edge['to_unit_id'][:8]}... | source: {edge['source']}"
        )


@app.command()
def neighbors(
    unit_id: str = typer.Argument(..., help="Unit ID"),
    depth: int = typer.Option(1, "--depth", "-d", help="Traversal depth (max 3)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Show a unit's neighborhood."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    depth = min(depth, 3)
    result = gs.get_neighbors(unit_id, depth=depth)

    if result["center"] is None:
        if json_output:
            _json_echo({"center": None, "neighbors": [], "edges": [], "error": "unit_not_found"})
            store.close()
            return
        typer.echo(f"Unit {unit_id} not found in graph.")
        store.close()
        return

    center_unit = store.get_unit(unit_id)
    if json_output:
        _json_echo(
            {
                "center": _unit_to_json(center_unit) if center_unit else None,
                "neighbors": [
                    _unit_to_json(unit)
                    for unit in (store.get_unit(nid) for nid in result["neighbors"])
                    if unit
                ],
                "edges": [_edge_to_json(edge) for edge in result["edges"]],
                "depth": depth,
            }
        )
        store.close()
        return

    typer.echo(f"Center: [{center_unit.source_project}] {center_unit.title}")
    typer.echo(f"  Neighbors ({len(result['neighbors'])}):")

    for nid in result["neighbors"]:
        n = store.get_unit(nid)
        if n:
            typer.echo(f"    [{n.source_project}] {n.title} ({n.id[:8]}...)")

    typer.echo(f"  Edges ({len(result['edges'])}):")
    for e in result["edges"]:
        typer.echo(f"    {e['from'][:8]}... --{e['relation']}--> {e['to'][:8]}...")

    store.close()


@app.command(name="shortest-path")
def shortest_path(
    from_id: str = typer.Argument(..., help="Start unit ID"),
    to_id: str = typer.Argument(..., help="End unit ID"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Show the shortest path between two units."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    missing = []
    if not gs.G.has_node(from_id):
        missing.append(f"source unit not found: {from_id}")
    if not gs.G.has_node(to_id):
        missing.append(f"target unit not found: {to_id}")
    if missing:
        if json_output:
            _json_echo({"path": [], "edges": [], "error": missing})
            store.close()
            return
        typer.echo("Error: " + "; ".join(missing) + ".")
        store.close()
        return

    path = gs.shortest_path(from_id, to_id)
    if path is None:
        if json_output:
            _json_echo({"path": [], "edges": [], "error": "no_path"})
            store.close()
            return
        typer.echo("No path found between the selected units.")
        store.close()
        return

    path_units = []
    for nid in path:
        unit = store.get_unit(nid)
        if unit:
            path_units.append(unit)

    if json_output:
        edges = []
        for left, right in zip(path_units, path_units[1:], strict=False):
            edge = gs.G.get_edge_data(left.id, right.id)
            direction = "forward"
            from_unit_id = left.id
            to_unit_id = right.id
            if edge is None:
                edge = gs.G.get_edge_data(right.id, left.id)
                direction = "reverse"
                from_unit_id = right.id
                to_unit_id = left.id
            edges.append(
                {
                    "id": edge.get("id") if edge else None,
                    "from_unit_id": from_unit_id,
                    "to_unit_id": to_unit_id,
                    "relation": str(edge.get("relation")) if edge else "related_to",
                    "weight": edge.get("weight") if edge else None,
                    "traversal_direction": direction,
                }
            )
        _json_echo(
            {
                "from_unit_id": from_id,
                "to_unit_id": to_id,
                "path": [_unit_to_json(unit) for unit in path_units],
                "edges": edges,
            }
        )
        store.close()
        return

    typer.echo(f"Shortest path ({len(path_units)} nodes):")
    for idx, unit in enumerate(path_units, 1):
        typer.echo(f"  {idx}. {_format_unit_label(unit)}")
        typer.echo(f"     ID: {unit.id}")

    if len(path_units) > 1:
        typer.echo("  Edges:")
        for left, right in zip(path_units, path_units[1:], strict=False):
            edge = gs.G.get_edge_data(left.id, right.id) or gs.G.get_edge_data(right.id, left.id)
            relation = edge.get("relation") if edge else "related_to"
            typer.echo(
                f"    {_format_unit_label(left)} --{relation}--> {_format_unit_label(right)}"
            )

    store.close()


@app.command()
def clusters(
    min_size: int = typer.Option(3, "--min-size", help="Minimum cluster size"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Find knowledge clusters."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    found = gs.get_clusters(min_size=min_size)
    if not found:
        if json_output:
            _json_echo({"results": []})
            store.close()
            return
        typer.echo("No clusters found.")
        store.close()
        return

    if json_output:
        _json_echo(
            {
                "results": [
                    {
                        "index": i,
                        "size": len(cluster),
                        "units": [
                            _unit_to_json(unit)
                            for unit in (store.get_unit(nid) for nid in cluster)
                            if unit
                        ],
                    }
                    for i, cluster in enumerate(found, 1)
                ]
            }
        )
        store.close()
        return

    for i, cluster in enumerate(found, 1):
        typer.echo(f"\nCluster {i} ({len(cluster)} nodes):")
        for nid in cluster[:5]:
            n = store.get_unit(nid)
            if n:
                typer.echo(f"  [{n.source_project}] {n.title}")
        if len(cluster) > 5:
            typer.echo(f"  ... and {len(cluster) - 5} more")

    store.close()


@app.command()
def gaps(
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Identify under-connected knowledge areas."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    found = gs.find_gaps()[:limit]
    if not found:
        if json_output:
            _json_echo({"results": []})
            store.close()
            return
        typer.echo("No gaps found.")
        store.close()
        return

    if json_output:
        results = []
        for gap in found:
            unit = store.get_unit(gap["unit_id"])
            results.append(
                {
                    "gap_type": gap["gap_type"],
                    "score": gap["score"],
                    "reason": gap["reason"],
                    "unit": _unit_to_json(unit) if unit else None,
                }
            )
        _json_echo({"results": results})
        store.close()
        return

    for g in found:
        unit = store.get_unit(g["unit_id"])
        if unit:
            typer.echo(
                f"[{g['gap_type']}] [{unit.source_project}] {unit.title} (score: {g['score']:.2f})"
            )
            typer.echo(f"  {g['reason']}")

    store.close()


@app.command()
def central(
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Find the most central knowledge units."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    found = gs.get_central_nodes(limit=limit)
    if not found:
        if json_output:
            _json_echo({"results": []})
            store.close()
            return
        typer.echo("No nodes found.")
        store.close()
        return

    if json_output:
        _json_echo(
            {
                "results": [
                    {"unit": _unit_to_json(unit), "score": score}
                    for unit, score in ((store.get_unit(nid), score) for nid, score in found)
                    if unit
                ]
            }
        )
        store.close()
        return

    for nid, score in found:
        unit = store.get_unit(nid)
        if unit:
            typer.echo(f"[{unit.source_project}] {unit.title} (PageRank: {score:.6f})")

    store.close()


@app.command()
def bridges(
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Find bridge knowledge units."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    found = gs.get_bridges(limit=limit)
    if not found:
        if json_output:
            _json_echo({"results": []})
            store.close()
            return
        typer.echo("No nodes found.")
        store.close()
        return

    if json_output:
        _json_echo(
            {
                "results": [
                    {"unit": _unit_to_json(unit), "score": score}
                    for unit, score in ((store.get_unit(nid), score) for nid, score in found)
                    if unit
                ]
            }
        )
        store.close()
        return

    for nid, score in found:
        unit = store.get_unit(nid)
        if unit:
            typer.echo(f"[{unit.source_project}] {unit.title} (betweenness: {score:.6f})")

    store.close()


@app.command(name="cross-project")
def cross_project(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Summarize cross-project connections."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    found = gs.cross_project_connections()
    if not found:
        if json_output:
            _json_echo({"results": []})
            store.close()
            return
        typer.echo("No cross-project connections found.")
        store.close()
        return

    if json_output:
        _json_echo({"results": found})
        store.close()
        return

    typer.echo("Cross-project connections:")
    for item in found:
        typer.echo(f"  {_format_project_pair(item['projects'])}: {item['edge_count']} edges")

    store.close()


@app.command(name="source-coverage")
def source_coverage(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Summarize coverage by source project and entity type."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    result = gs.analyze_source_coverage()

    if json_output:
        _json_echo(result)
        store.close()
        return

    sources = result["sources"]
    if not sources:
        typer.echo("No sources found.")
        store.close()
        return

    typer.echo("Source coverage:")
    for item in sources:
        typer.echo(f"  {item['source_project']}/{item['source_entity_type']}")
        typer.echo(
            f"    Units: {item['unit_count']} | Edges: {item['edge_count']} | "
            f"Orphans: {item['orphan_count']}"
        )
        typer.echo(
            f"    Created: {item['oldest_created_at'] or '-'} -> {item['newest_created_at'] or '-'}"
        )
        if item["has_sync_state"]:
            typer.echo(f"    Sync: {item['last_sync_at']} | Items synced: {item['items_synced']}")
        else:
            typer.echo("    Sync: -")

    store.close()


@app.command()
def timeline(
    bucket: str = typer.Option(
        "month",
        "--bucket",
        help="Bucket size: day | week | month | year",
    ),
    field: str = typer.Option(
        "created_at",
        "--field",
        help="Timestamp field: created_at | ingested_at | updated_at",
    ),
    start: str | None = typer.Option(None, "--start", help="ISO-8601 inclusive start"),
    end: str | None = typer.Option(None, "--end", help="ISO-8601 inclusive end"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max buckets to return"),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    tag: str | None = typer.Option(None, "--tag", help="Require an exact graph tag"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Analyze knowledge growth over time."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    try:
        result = gs.analyze_timeline(
            bucket=bucket,
            field=field,
            start=start,
            end=end,
            limit=limit,
            source_project=source_project,
            content_type=content_type,
            tag=tag,
        )
    except ValueError as exc:
        store.close()
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        _json_echo(result)
        store.close()
        return

    typer.echo(f"Timeline: {result['total']} units by {result['bucket']} using {result['field']}")
    if not result["buckets"]:
        typer.echo("No timeline buckets found.")
        store.close()
        return

    for item in result["buckets"]:
        projects = ", ".join(
            f"{project}:{count}" for project, count in item["source_projects"].items()
        )
        content_types = ", ".join(
            f"{found_content_type}:{count}"
            for found_content_type, count in item["content_types"].items()
        )
        tags = ", ".join(
            f"{found_tag['tag']}:{found_tag['count']}" for found_tag in item["top_tags"]
        )
        typer.echo(f"{item['bucket']}: {item['count']} units")
        typer.echo(f"  Projects: {projects or '-'}")
        typer.echo(f"  Content types: {content_types or '-'}")
        typer.echo(f"  Top tags: {tags or '-'}")

    store.close()


@tags_app.callback(invoke_without_command=True)
def tags(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", "-n", help="Max tags or matching units"),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    tag: str | None = typer.Option(None, "--tag", help="Show detail for one exact tag"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Explore tag taxonomy and co-occurring topics."""
    if ctx.invoked_subcommand is not None:
        return

    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    result = gs.analyze_tags(
        tag=tag,
        limit=limit,
        source_project=source_project,
        content_type=content_type,
    )

    if json_output:
        _json_echo(result)
        store.close()
        return

    if tag:
        typer.echo(f"Tag: {result['tag']} ({result['count']} units)")
        typer.echo("By source project:")
        for project, count in result["source_projects"].items():
            typer.echo(f"  {project}: {count}")
        typer.echo("By content type:")
        for found_content_type, count in result["content_types"].items():
            typer.echo(f"  {found_content_type}: {count}")

        co_occurring_tags = result["co_occurring_tags"]
        if co_occurring_tags:
            typer.echo("Co-occurring tags:")
            for item in co_occurring_tags:
                typer.echo(f"  {item['tag']}: {item['count']}")

        units = result["units"]
        if units:
            typer.echo("Matching units:")
            for unit in units:
                typer.echo(f"  [{unit['source_project']}] {unit['title']}")
                typer.echo(
                    f"    ID: {unit['id']} | Type: {unit['content_type']} | "
                    f"Tags: {', '.join(unit['tags'])}"
                )
        store.close()
        return

    found_tags = result["tags"]
    if not found_tags:
        typer.echo("No tags found.")
        store.close()
        return

    typer.echo("Top tags:")
    for item in found_tags:
        projects = ", ".join(
            f"{project}:{count}" for project, count in item["source_projects"].items()
        )
        content_types = ", ".join(
            f"{found_content_type}:{count}"
            for found_content_type, count in item["content_types"].items()
        )
        typer.echo(f"  {item['tag']}: {item['count']}")
        typer.echo(f"    Projects: {projects or '-'}")
        typer.echo(f"    Content types: {content_types or '-'}")

    store.close()


@app.command(name="tag-graph")
def tag_graph(
    limit: int = typer.Option(20, "--limit", "-n", help="Max tag pairs to return"),
    min_count: int = typer.Option(
        1,
        "--min-count",
        help="Minimum co-occurrence count required for a tag pair",
    ),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Analyze tag co-occurrence as a weighted graph."""
    store = _get_store()
    try:
        result = _do_tag_graph(
            store,
            source_project=source_project,
            content_type=content_type,
            min_count=min_count,
            limit=limit,
        )
    except ValueError as exc:
        store.close()
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        _json_echo(result)
        store.close()
        return

    edges = result["edges"]
    if not edges:
        typer.echo("No tag co-occurrences found.")
        store.close()
        return

    typer.echo("Top tag pairs:")
    for edge in edges:
        representatives = ", ".join(edge["representative_unit_ids"])
        typer.echo(
            f"  {edge['source']} <-> {edge['target']}: "
            f"{edge['co_occurrence_count']} units"
        )
        typer.echo(f"    Representative units: {representatives or '-'}")

    store.close()


@tags_app.command(name="apply-search")
def tags_apply_search(
    query: str = typer.Argument(..., help="Search query"),
    add_tags: list[str] = typer.Option(
        [],
        "--add",
        help="Tag to add to matching units. Repeat to add multiple tags.",
    ),
    remove_tags: list[str] = typer.Option(
        [],
        "--remove",
        help="Tag to remove from matching units. Repeat to remove multiple tags.",
    ),
    mode: str = typer.Option(
        "fulltext", "--mode", "-m", help="Search mode: fulltext | semantic | hybrid"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Max matching units to update"),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    tag: str | None = typer.Option(None, "--tag", help="Require an exact existing tag"),
    review_state: str | None = typer.Option(
        None,
        "--review-state",
        help="Filter by review state metadata",
    ),
    created_after: str | None = typer.Option(
        None,
        "--created-after",
        help="Filter to units created on or after an ISO-8601 date/datetime",
    ),
    created_before: str | None = typer.Option(
        None,
        "--created-before",
        help="Filter to units created on or before an ISO-8601 date/datetime",
    ),
    updated_after: str | None = typer.Option(
        None,
        "--updated-after",
        help="Filter to units updated on or after an ISO-8601 date/datetime",
    ),
    updated_before: str | None = typer.Option(
        None,
        "--updated-before",
        help="Filter to units updated on or before an ISO-8601 date/datetime",
    ),
    min_utility: float | None = typer.Option(
        None,
        "--min-utility",
        help="Filter to units with utility score at least this value",
    ),
    max_utility: float | None = typer.Option(
        None,
        "--max-utility",
        help="Filter to units with utility score at most this value",
    ),
    min_confidence: float | None = typer.Option(
        None,
        "--min-confidence",
        help="Filter to units with confidence at least this value",
    ),
    max_confidence: float | None = typer.Option(
        None,
        "--max-confidence",
        help="Filter to units with confidence at most this value",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without writing"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Apply tag additions/removals to units matched by a search."""
    store = _get_store()
    filters = _search_filters_dict(
        source_project=source_project,
        content_type=content_type,
        tag=tag,
        review_state=review_state,
        created_after=created_after,
        created_before=created_before,
        updated_after=updated_after,
        updated_before=updated_before,
        min_utility=min_utility,
        max_utility=max_utility,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )
    try:
        result = _do_apply_tags_to_search(
            store,
            query,
            add_tags=add_tags,
            remove_tags=remove_tags,
            limit=limit,
            mode=mode,
            filters=filters,
            dry_run=dry_run,
        )
    except ValueError as exc:
        store.close()
        if json_output:
            _json_echo(
                {
                    "query": query,
                    "mode": mode,
                    "limit": limit,
                    "filters": filters,
                    "add_tags": add_tags,
                    "remove_tags": remove_tags,
                    "dry_run": dry_run,
                    "matched_count": 0,
                    "changed_count": 0,
                    "affected_count": 0,
                    "changed_units": [],
                    "affected_units": [],
                    "error": str(exc),
                }
            )
            return
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        _json_echo(result)
        store.close()
        return

    action = "Would update" if dry_run else "Updated"
    typer.echo(f"{action} {result['changed_count']} of {result['matched_count']} matched units.")
    for unit in result["changed_units"]:
        typer.echo(f"  [{unit['source_project']}] {unit['title']}")
        typer.echo(
            f"    ID: {unit['id']} | Tags: {', '.join(unit['old_tags'])} -> "
            f"{', '.join(unit['new_tags'])}"
        )
    store.close()


@app.command(name="tag-synonyms")
def tag_synonyms(
    limit: int = typer.Option(20, "--limit", "-n", help="Max synonym groups"),
    min_similarity: float = typer.Option(
        0.8,
        "--min-similarity",
        help="Minimum normalized character similarity",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Suggest likely synonymous or variant tags without modifying data."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    result = gs.suggest_tag_synonyms(limit=limit, min_similarity=min_similarity)

    if json_output:
        _json_echo(result)
        store.close()
        return

    suggestions = result["suggestions"]
    if not suggestions:
        typer.echo("No tag synonym suggestions found.")
        store.close()
        return

    typer.echo("Tag synonym suggestions:")
    for suggestion in suggestions:
        typer.echo(
            f"  {suggestion['canonical_candidate']} "
            f"({suggestion['total_count']} uses, "
            f"{suggestion['variant_count']} variants, "
            f"similarity {suggestion['similarity']:.3f})"
        )
        for variant in suggestion["variants"]:
            typer.echo(f"    {variant['tag']}: {variant['count']}")

    store.close()


@app.command(name="rename-tag")
def rename_tag(
    old_tag: str = typer.Argument(..., help="Exact tag to replace"),
    new_tag: str = typer.Argument(..., help="Replacement tag"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without writing"),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Rename or merge one exact tag across matching units."""
    store = _get_store()
    try:
        result = _do_rename_tag(
            store,
            old_tag,
            new_tag,
            dry_run=dry_run,
            source_project=source_project,
            content_type=content_type,
        )
    except ValueError as exc:
        store.close()
        if json_output:
            _json_echo(
                {
                    "old_tag": old_tag,
                    "new_tag": new_tag,
                    "dry_run": dry_run,
                    "changed_count": 0,
                    "changed_units": [],
                    "sample_units": [],
                    "error": str(exc),
                }
            )
            return
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        _json_echo(result)
        store.close()
        return

    action = "Would update" if dry_run else "Updated"
    typer.echo(
        f"{action} {result['changed_count']} units: {result['old_tag']} -> {result['new_tag']}"
    )
    for unit in result["sample_units"]:
        typer.echo(f"  [{unit['source_project']}] {unit['title']}")
        typer.echo(f"    ID: {unit['id']} | Tags: {', '.join(unit['new_tags'])}")
    store.close()


@app.command(name="links")
def links(
    domain: str | None = typer.Option(None, "--domain", help="Filter by exact domain"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max domains and URLs"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Inventory external http/https links across content and metadata."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    result = gs.analyze_links(domain=domain, limit=limit)

    if json_output:
        _json_echo(result)
        store.close()
        return

    found_domains = result["domains"]
    if not found_domains:
        typer.echo("No external links found.")
        store.close()
        return

    typer.echo("Top external link domains:")
    for item in found_domains:
        typer.echo(
            f"  {item['domain']}: {item['count']} occurrences across {item['url_count']} URLs"
        )
        units = ", ".join(
            f"[{unit['source_project']}] {unit['title']}"
            for unit in item["representative_units"][:3]
        )
        typer.echo(f"    Units: {units or '-'}")
        for link in item["urls"][:3]:
            typer.echo(f"    {link['url']} ({link['count']})")

    store.close()


@app.command(name="suggest-edges")
def suggest_edges(
    limit: int = typer.Option(20, "--limit", "-n", help="Max candidate edges"),
    min_score: float = typer.Option(0.4, "--min-score", help="Minimum suggestion score"),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter candidate units by source project",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Suggest likely missing edges without writing relationships."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    result = gs.suggest_edges(
        limit=limit,
        min_score=min_score,
        source_project=source_project,
    )

    if json_output:
        _json_echo(result)
        store.close()
        return

    candidates = result["candidates"]
    if not candidates:
        typer.echo("No edge suggestions found.")
        store.close()
        return

    typer.echo("Edge suggestions:")
    for candidate in candidates:
        from_unit = candidate["from_unit"]
        to_unit = candidate["to_unit"]
        typer.echo(f"  {from_unit['title']} -> {to_unit['title']} score={candidate['score']:.3f}")
        typer.echo(f"    IDs: {candidate['from_id']} -> {candidate['to_id']}")
        for reason in candidate["reasons"]:
            typer.echo(f"    - {reason}")

    store.close()


@app.command(name="duplicates")
def duplicates(
    limit: int = typer.Option(20, "--limit", "-n", help="Max duplicate groups"),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Find likely duplicate knowledge units."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    result = gs.analyze_duplicates(
        limit=limit,
        source_project=source_project,
        content_type=content_type,
    )

    if json_output:
        _json_echo(result)
        store.close()
        return

    found = result["results"]
    if not found:
        typer.echo("No duplicates found.")
        store.close()
        return

    for item in found:
        typer.echo(f"[{item['reason']}] score: {item['score']:.3f}")
        for unit in item["units"]:
            typer.echo(f"  [{unit['source_project']}] {unit['title']} ({unit['content_type']})")
            typer.echo(f"    ID: {unit['id']}")

    store.close()


@app.command(name="review-queue")
def review_queue(
    limit: int = typer.Option(20, "--limit", "-n", help="Max queue items"),
    source_project: str | None = typer.Option(
        None,
        "--source-project",
        "--project",
        "-p",
        help="Filter by source project",
    ),
    content_type: str | None = typer.Option(None, "--content-type", help="Filter by content type"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Rank knowledge units worth revisiting."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    result = gs.build_review_queue(
        limit=limit,
        source_project=source_project,
        content_type=content_type,
    )

    if json_output:
        _json_echo(result)
        store.close()
        return

    queue = result["queue"]
    if not queue:
        typer.echo("No review candidates found.")
        store.close()
        return

    typer.echo("Review queue:")
    for item in queue:
        unit = item["unit"]
        reasons = ", ".join(f"{reason['code']}:{reason['score']:.1f}" for reason in item["reasons"])
        typer.echo(f"  [{unit['source_project']}] {unit['title']} score={item['score']:.1f}")
        typer.echo(
            f"    ID: {unit['id']} | Type: {unit['content_type']} | "
            f"Degree: {item['degree']} | Age: {item['age_days']}d"
        )
        typer.echo(f"    Reasons: {reasons}")

    store.close()


def _do_export_obsidian(
    store: Store,
    vault_path: str | None = None,
    folder: str = "Graph",
    clean: bool = False,
) -> int:
    """Core Obsidian export logic. Returns number of notes written."""
    import re
    import shutil
    from pathlib import Path

    vault_path = vault_path or settings.obsidian_vault_path
    output_dir = Path(vault_path) / folder
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    # Create directories for all source projects dynamically
    all_units = store.get_all_units()
    for u in all_units:
        (output_dir / u.source_project).mkdir(parents=True, exist_ok=True)
    all_edges = store.get_all_edges()

    unit_map = {u.id: u for u in all_units}

    outgoing: dict[str, list[tuple[str, str]]] = {}
    incoming: dict[str, list[tuple[str, str]]] = {}
    for e in all_edges:
        outgoing.setdefault(e.from_unit_id, []).append((e.relation, e.to_unit_id))
        incoming.setdefault(e.to_unit_id, []).append((e.relation, e.from_unit_id))

    def _safe_filename(title: str) -> str:
        name = re.sub(r'[<>:"/\\|?*]', "", title)
        name = name.strip(". ")
        return name[:120] if name else "Untitled"

    written = 0
    for u in all_units:
        filename = _safe_filename(u.title)
        filepath = output_dir / u.source_project / f"{filename}.md"

        lines = []

        lines.append("---")
        lines.append(f"id: {u.id}")
        lines.append(f"source_project: {u.source_project}")
        lines.append(f"source_entity_type: {u.source_entity_type}")
        lines.append(f"content_type: {u.content_type}")
        if u.tags:
            lines.append(f"tags: [{', '.join(u.tags)}]")
        if u.confidence is not None:
            lines.append(f"confidence: {u.confidence}")
        if u.utility_score is not None:
            lines.append(f"utility_score: {u.utility_score}")
        lines.append(f"created_at: {u.created_at}")
        lines.append("---")
        lines.append("")

        lines.append(u.content)
        lines.append("")

        out_edges = outgoing.get(u.id, [])
        if out_edges:
            lines.append("## Connections")
            lines.append("")
            for relation, target_id in out_edges:
                target = unit_map.get(target_id)
                if target:
                    link_path = f"{folder}/{target.source_project}/{_safe_filename(target.title)}"
                    lines.append(f"- **{relation}** [[{link_path}|{target.title}]]")
            lines.append("")

        in_edges = incoming.get(u.id, [])
        if in_edges:
            lines.append("## Referenced by")
            lines.append("")
            for relation, source_id in in_edges:
                source = unit_map.get(source_id)
                if source:
                    link_path = f"{folder}/{source.source_project}/{_safe_filename(source.title)}"
                    lines.append(f"- **{relation}** from [[{link_path}|{source.title}]]")
            lines.append("")

        if u.metadata:
            interesting = {k: v for k, v in u.metadata.items() if v}
            if interesting:
                lines.append("## Metadata")
                lines.append("")
                for k, v in interesting.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"- **{k}**: `{json.dumps(v)}`")
                    else:
                        lines.append(f"- **{k}**: {v}")
                lines.append("")

        filepath.write_text("\n".join(lines))
        written += 1

    index_path = output_dir / "Index.md"
    index_lines = [
        "# Knowledge Graph Index",
        "",
        f"**{len(all_units)} units** | **{len(all_edges)} edges**",
        "",
    ]
    for proj in ["forty_two", "max", "presence", "me", "bookmarks"]:
        proj_units = [u for u in all_units if u.source_project == proj]
        if proj_units:
            index_lines.append(f"## {proj} ({len(proj_units)})")
            index_lines.append("")
            for u in sorted(proj_units, key=lambda x: x.title):
                link_path = f"{folder}/{u.source_project}/{_safe_filename(u.title)}"
                index_lines.append(f"- [[{link_path}|{u.title}]] `{u.content_type}`")
            index_lines.append("")

    index_path.write_text("\n".join(index_lines))

    typer.echo(f"Exported {written} notes + index to {output_dir}")
    return written


@app.command(name="export-obsidian")
def export_obsidian(
    vault_path: str | None = typer.Option(
        None,
        "--vault",
        "-v",
        help="Path to Obsidian vault",
    ),
    folder: str = typer.Option("Graph", "--folder", "-f", help="Subfolder within vault"),
    clean: bool = typer.Option(False, "--clean", help="Remove existing folder before export"),
) -> None:
    """Export knowledge graph to Obsidian vault as markdown notes with wikilinks."""
    store = _get_store()
    _do_export_obsidian(store, vault_path=vault_path, folder=folder, clean=clean)
    store.close()


@app.command()
def sync(
    vault_path: str | None = typer.Option(
        None,
        "--vault",
        "-v",
        help="Path to Obsidian vault",
    ),
    batch_size: int = typer.Option(5, "--batch-size", "-b", help="Embedding batch size"),
    delay: float = typer.Option(21.0, "--delay", help="Seconds between embedding batches"),
    full_ingest: bool = typer.Option(
        False,
        "--full-ingest",
        help="Ignore sync state during the ingest step",
    ),
) -> None:
    """Full sync pipeline: ingest -> embed -> export to Obsidian."""
    store = _get_store()

    typer.echo("=== Graph Sync ===\n")

    typer.echo("Step 1/3: Ingesting from all sources...")
    _do_ingest(store, project="all", full=full_ingest)

    typer.echo("\nStep 2/3: Embedding new units...")
    _do_embed(store, batch_size=batch_size, delay=delay)

    typer.echo("\nStep 3/3: Exporting to Obsidian...")
    _do_export_obsidian(store, vault_path=vault_path, clean=True)

    store.close()
    typer.echo("\n=== Sync complete ===")


if __name__ == "__main__":
    app()
