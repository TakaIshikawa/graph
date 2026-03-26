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
    }
    factory = mapping.get(name)
    if factory is None:
        raise typer.BadParameter(f"Unknown project: {name}. Available: {list(mapping)}")
    return factory()


@app.command()
def ingest(
    project: str = typer.Argument("all", help="Source project or 'all'"),
    entity_type: str | None = typer.Option(None, "--type", "-t", help="Specific entity type"),
) -> None:
    """Ingest knowledge from source projects."""
    store = _get_store()

    projects = ["forty_two", "max", "presence", "me"] if project == "all" else [project]
    entity_types = [entity_type] if entity_type else None

    total_stats = {"units_inserted": 0, "units_skipped": 0, "edges_inserted": 0}

    for proj in projects:
        adapter = _get_adapter_for_project(proj)
        since = None
        for et in adapter.entity_types:
            s = store.get_sync_state(proj, et)
            if s and (since is None or s.last_sync_at < since.last_sync_at):
                since = s

        typer.echo(f"Ingesting from {proj}...")
        result = adapter.ingest(since=since, entity_types=entity_types)
        stats = store.ingest(result, proj)

        for key in total_stats:
            total_stats[key] += stats[key]

        typer.echo(
            f"  {proj}: {stats['units_inserted']} new, "
            f"{stats['units_skipped']} updated, "
            f"{stats['edges_inserted']} edges"
        )

        # Update sync state per entity type
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

    store.close()
    typer.echo(
        f"\nTotal: {total_stats['units_inserted']} new units, "
        f"{total_stats['units_skipped']} updated, "
        f"{total_stats['edges_inserted']} edges"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    mode: str = typer.Option("fulltext", "--mode", "-m", help="Search mode: fulltext | semantic | hybrid"),
    project: str | None = typer.Option(None, "--project", "-p", help="Filter by source project"),
) -> None:
    """Search knowledge units."""
    store = _get_store()

    if mode == "fulltext":
        results = store.fts_search(query, limit=limit)
        if not results:
            typer.echo("No results found.")
            store.close()
            return
        for r in results:
            unit = store.get_unit(r["unit_id"])
            if unit and (project is None or unit.source_project == project):
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
            pairs = rag.search(
                query, limit=limit, source_project=project, min_similarity=0.3
            )
        else:
            pairs = rag.hybrid_search(query, limit=limit)

        if not pairs:
            typer.echo("No results found.")
            store.close()
            return

        for unit, score in pairs:
            if project and unit.source_project != project:
                continue
            typer.echo(f"\n[{unit.source_project}] {unit.title}  (score: {score:.3f})")
            typer.echo(f"  ID: {unit.id}")
            typer.echo(f"  Type: {unit.content_type} | Tags: {', '.join(unit.tags)}")
            typer.echo(f"  {unit.content[:120]}...")
    else:
        typer.echo(f"Unknown mode: {mode}. Use fulltext, semantic, or hybrid.")

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
    typer.echo(f"\nBy project:")
    for proj, count in s["by_project"].items():
        typer.echo(f"  {proj}: {count}")
    typer.echo(f"\nBy content type:")
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

    projects = ["forty_two", "max", "presence", "me"]
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


if __name__ == "__main__":
    app()
