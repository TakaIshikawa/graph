"""CLI for the Graph personal knowledge graph."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import typer

from graph.config import settings
from graph.store.db import Store
from graph.types.models import SyncState

app = typer.Typer(name="graph", help="Personal Knowledge Graph — aggregate, connect, retrieve")


def _get_store() -> Store:
    return Store(settings.database_url)


def _get_adapter_for_project(name: str):
    from graph.adapters.forty_two import FortyTwoAdapter
    from graph.adapters.kindle import KindleAdapter
    from graph.adapters.max_adapter import MaxAdapter
    from graph.adapters.me import MeAdapter
    from graph.adapters.presence import PresenceAdapter
    from graph.adapters.sota import SOTAAdapter

    mapping = {
        "forty_two": lambda: FortyTwoAdapter(db_path=settings.forty_two_db),
        "max": lambda: MaxAdapter(db_path=settings.max_db),
        "presence": lambda: PresenceAdapter(
            db_path=settings.presence_db, min_score=settings.content_min_score
        ),
        "me": lambda: MeAdapter(config_path=settings.me_config),
        "kindle": lambda: KindleAdapter(db_path=settings.kindle_db),
        "sota": lambda: SOTAAdapter(db_path=settings.sota_db),
    }
    factory = mapping.get(name)
    if factory is None:
        raise typer.BadParameter(f"Unknown project: {name}. Available: {list(mapping)}")
    return factory()


def _format_unit_label(unit) -> str:
    return f"[{unit.source_project}] {unit.title}"


def _format_project_pair(projects: list[str]) -> str:
    return f"{projects[0]} <-> {projects[1]}"


def _unit_matches_search_filters(
    unit,
    *,
    source_project: str | None = None,
    content_type: str | None = None,
    tag: str | None = None,
    review_state: str | None = None,
) -> bool:
    if source_project and str(unit.source_project) != source_project:
        return False
    if content_type and str(unit.content_type) != content_type:
        return False
    if tag and tag not in unit.tags:
        return False
    if review_state and unit.metadata.get("review_state") != review_state:
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
) -> list:
    fetch_limit = max(limit, 20)

    while True:
        results = store.fts_search(query, limit=fetch_limit)
        filtered = []
        for r in results:
            unit = store.get_unit(r["unit_id"])
            if unit and _unit_matches_search_filters(
                unit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                review_state=review_state,
            ):
                filtered.append(unit)
                if len(filtered) >= limit:
                    return filtered

        if len(results) < fetch_limit:
            return filtered

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
) -> list[tuple[object, float]]:
    fetch_limit = max(limit, 20)

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
            ):
                filtered.append((unit, score))
                if len(filtered) >= limit:
                    return filtered

        if len(pairs) < fetch_limit:
            return filtered

        fetch_limit *= 2


def _do_ingest(
    store: Store,
    project: str = "all",
    entity_type: str | None = None,
    full: bool = False,
) -> dict:
    """Core ingest logic. Returns total stats dict."""
    projects = (
        ["forty_two", "max", "presence", "me", "kindle", "sota"]
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


def _do_embed(
    store: Store,
    project: str | None = None,
    batch_size: int = 5,
    delay: float = 21.0,
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

    all_units = store.get_all_units()
    embedded_ids = {unit.id for unit, _ in store.get_units_with_embeddings()}

    units_to_embed = [
        u.id
        for u in all_units
        if u.id not in embedded_ids and (project is None or u.source_project == project)
    ]

    if not units_to_embed:
        typer.echo("All units already have embeddings.")
        return 0

    total_batches = (len(units_to_embed) + batch_size - 1) // batch_size
    typer.echo(f"Embedding {len(units_to_embed)} units in {total_batches} batches...")
    total = 0
    for i in range(0, len(units_to_embed), batch_size):
        batch = units_to_embed[i : i + batch_size]
        count = rag.embed_batch_and_store(batch)
        total += count
        batch_num = i // batch_size + 1
        typer.echo(f"  Batch {batch_num}/{total_batches}: {count} embedded ({total}/{len(units_to_embed)})")
        if i + batch_size < len(units_to_embed):
            time.sleep(delay)

    typer.echo(f"Done. {total} units embedded.")
    return total


@app.command()
def embed(
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by source project"),
    batch_size: int = typer.Option(5, "--batch-size", "-b", help="Batch size for API calls"),
    delay: float = typer.Option(21.0, "--delay", help="Seconds between batches (rate limit)"),
) -> None:
    """Generate embeddings for all units missing them."""
    store = _get_store()
    _do_embed(store, project=project, batch_size=batch_size, delay=delay)
    store.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    mode: str = typer.Option("fulltext", "--mode", "-m", help="Search mode: fulltext | semantic | hybrid"),
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
) -> None:
    """Search knowledge units."""
    store = _get_store()

    if mode == "fulltext":
        results = _search_fulltext_with_filters(
            store,
            query,
            limit=limit,
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            review_state=review_state,
        )
        if not results:
            typer.echo("No results found.")
            store.close()
            return
        for unit in results:
            typer.echo(f"\n[{unit.source_project}] {unit.title}")
            typer.echo(f"  ID: {unit.id}")
            typer.echo(f"  Type: {unit.content_type} | Tags: {', '.join(unit.tags)}")
            typer.echo(f"  {unit.content[:120]}...")
    elif mode in ("semantic", "hybrid"):
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
                lambda q, fetch_limit: rag.search(q, limit=fetch_limit, min_similarity=0.3),
                query,
                limit=limit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                review_state=review_state,
            )
        else:
            pairs = _search_scored_with_filters(
                lambda q, fetch_limit: rag.hybrid_search(q, limit=fetch_limit),
                query,
                limit=limit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                review_state=review_state,
            )

        if not pairs:
            typer.echo("No results found.")
            store.close()
            return

        for unit, score in pairs:
            typer.echo(f"\n[{unit.source_project}] {unit.title}  (score: {score:.3f})")
            typer.echo(f"  ID: {unit.id}")
            typer.echo(f"  Type: {unit.content_type} | Tags: {', '.join(unit.tags)}")
            typer.echo(f"  {unit.content[:120]}...")
    else:
        typer.echo(f"Unknown mode: {mode}. Use fulltext, semantic, or hybrid.")

    store.close()


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
        typer.echo("No matching ideas found.")
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


@app.command()
def stats() -> None:
    """Show graph statistics."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()
    s = gs.stats()

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
def serve() -> None:
    """Start the MCP server (stdio transport)."""
    import asyncio

    from graph.mcp.server import run_mcp_server

    asyncio.run(run_mcp_server())


@app.command(name="sync-status")
def sync_status() -> None:
    """Show last sync timestamps per source project."""
    store = _get_store()

    projects = ["forty_two", "max", "presence", "me", "sota"]
    for proj in projects:
        adapter = _get_adapter_for_project(proj)
        typer.echo(f"\n{proj}:")
        for et in adapter.entity_types:
            state = store.get_sync_state(proj, et)
            if state:
                typer.echo(f"  {et}: last sync {state.last_sync_at} ({state.items_synced} total)")
            else:
                typer.echo(f"  {et}: never synced")

    store.close()


@app.command()
def neighbors(
    unit_id: str = typer.Argument(..., help="Unit ID"),
    depth: int = typer.Option(1, "--depth", "-d", help="Traversal depth (max 3)"),
) -> None:
    """Show a unit's neighborhood."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    depth = min(depth, 3)
    result = gs.get_neighbors(unit_id, depth=depth)

    if result["center"] is None:
        typer.echo(f"Unit {unit_id} not found in graph.")
        store.close()
        return

    center_unit = store.get_unit(unit_id)
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
        typer.echo("Error: " + "; ".join(missing) + ".")
        store.close()
        return

    path = gs.shortest_path(from_id, to_id)
    if path is None:
        typer.echo("No path found between the selected units.")
        store.close()
        return

    path_units = []
    for nid in path:
        unit = store.get_unit(nid)
        if unit:
            path_units.append(unit)

    typer.echo(f"Shortest path ({len(path_units)} nodes):")
    for idx, unit in enumerate(path_units, 1):
        typer.echo(f"  {idx}. {_format_unit_label(unit)}")
        typer.echo(f"     ID: {unit.id}")

    if len(path_units) > 1:
        typer.echo("  Edges:")
        for left, right in zip(path_units, path_units[1:], strict=False):
            edge = gs.G.get_edge_data(left.id, right.id) or gs.G.get_edge_data(
                right.id, left.id
            )
            relation = edge.get("relation") if edge else "related_to"
            typer.echo(
                f"    {_format_unit_label(left)} --{relation}--> "
                f"{_format_unit_label(right)}"
            )

    store.close()


@app.command()
def clusters(
    min_size: int = typer.Option(3, "--min-size", help="Minimum cluster size"),
) -> None:
    """Find knowledge clusters."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    found = gs.get_clusters(min_size=min_size)
    if not found:
        typer.echo("No clusters found.")
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
) -> None:
    """Identify under-connected knowledge areas."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    found = gs.find_gaps()[:limit]
    if not found:
        typer.echo("No gaps found.")
        store.close()
        return

    for g in found:
        unit = store.get_unit(g["unit_id"])
        if unit:
            typer.echo(
                f"[{g['gap_type']}] [{unit.source_project}] {unit.title} "
                f"(score: {g['score']:.2f})"
            )
            typer.echo(f"  {g['reason']}")

    store.close()


@app.command()
def central(
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
) -> None:
    """Find the most central knowledge units."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    found = gs.get_central_nodes(limit=limit)
    if not found:
        typer.echo("No nodes found.")
        store.close()
        return

    for nid, score in found:
        unit = store.get_unit(nid)
        if unit:
            typer.echo(
                f"[{unit.source_project}] {unit.title} "
                f"(PageRank: {score:.6f})"
            )

    store.close()


@app.command()
def bridges(
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
) -> None:
    """Find bridge knowledge units."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    found = gs.get_bridges(limit=limit)
    if not found:
        typer.echo("No nodes found.")
        store.close()
        return

    for nid, score in found:
        unit = store.get_unit(nid)
        if unit:
            typer.echo(
                f"[{unit.source_project}] {unit.title} "
                f"(betweenness: {score:.6f})"
            )

    store.close()


@app.command(name="cross-project")
def cross_project() -> None:
    """Summarize cross-project connections."""
    from graph.graph.service import GraphService

    store = _get_store()
    gs = GraphService(store)
    gs.rebuild()

    found = gs.cross_project_connections()
    if not found:
        typer.echo("No cross-project connections found.")
        store.close()
        return

    typer.echo("Cross-project connections:")
    for item in found:
        typer.echo(
            f"  {_format_project_pair(item['projects'])}: {item['edge_count']} edges"
        )

    store.close()


def _do_export_obsidian(
    store: Store,
    vault_path: str = "/Users/taka/ObsidianVaults/note",
    folder: str = "Graph",
    clean: bool = False,
) -> int:
    """Core Obsidian export logic. Returns number of notes written."""
    import re
    import shutil
    from pathlib import Path

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
        name = re.sub(r'[<>:"/\\|?*]', '', title)
        name = name.strip('. ')
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
    for proj in ["forty_two", "max", "presence", "me"]:
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
    vault_path: str = typer.Option(
        "/Users/taka/ObsidianVaults/note",
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
    vault_path: str = typer.Option(
        "/Users/taka/ObsidianVaults/note",
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
