"""CSV export for weakly connected edge components."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["component_id", "node_count", "edge_count", "source_nodes", "sink_nodes", "representative_nodes", "density"]


def export_edge_weak_component_csv(
    edges: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    nodes: Iterable[Mapping[str, Any] | object] | None = None,
) -> str | dict[str, Any]:
    """Return or write weakly connected component metrics for directed edges."""
    edge_list = list(edges)
    node_ids = {_source(edge) for edge in edge_list} | {_target(edge) for edge in edge_list}
    if nodes is not None:
        node_ids.update(unit_id(node) or field_value(node) for node in nodes)
    node_ids.discard("")
    rows = _rows(edge_list, node_ids)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "component_count": len(rows), "edge_count": len(edge_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(edges: list[Mapping[str, Any] | object], node_ids: set[str]) -> list[dict[str, str | int]]:
    undirected: dict[str, set[str]] = {node: set() for node in node_ids}
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source = _source(edge)
        target = _target(edge)
        if not source or not target:
            continue
        undirected.setdefault(source, set()).add(target)
        undirected.setdefault(target, set()).add(source)
        outgoing[source].add(target)
        incoming[target].add(source)

    components = []
    seen: set[str] = set()
    for start in sorted(undirected, key=sort_key):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component: set[str] = set()
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in sorted(undirected[node], key=sort_key):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    rows = []
    for component in sorted(components, key=lambda nodes: sort_key(min(nodes, key=sort_key))):
        component_edges = [edge for edge in edges if _source(edge) in component and _target(edge) in component]
        sources = sorted((node for node in component if outgoing[node] and not incoming[node]), key=sort_key)
        sinks = sorted((node for node in component if incoming[node] and not outgoing[node]), key=sort_key)
        representatives = sorted(component, key=sort_key)[:3]
        node_count = len(component)
        possible_edges = node_count * (node_count - 1)
        rows.append(
            {
                "component_id": min(component, key=sort_key),
                "node_count": node_count,
                "edge_count": len(component_edges),
                "source_nodes": "; ".join(sources),
                "sink_nodes": "; ".join(sinks),
                "representative_nodes": "; ".join(representatives),
                "density": f"{(len(component_edges) / possible_edges if possible_edges else 0):.4f}",
            }
        )
    return rows


def _source(edge: Mapping[str, Any] | object) -> str:
    return field_value(get(edge, "source_id") or get(edge, "from_unit_id") or get(edge, "source_unit_id"))


def _target(edge: Mapping[str, Any] | object) -> str:
    return field_value(get(edge, "target_id") or get(edge, "to_unit_id") or get(edge, "target_unit_id"))
