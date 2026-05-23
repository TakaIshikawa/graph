"""CSV export for edge directionality patterns by edge type."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, write_csv
from graph.types.models import KnowledgeEdge

_FIELDNAMES = ["edge_type", "total_edges", "reciprocal_pairs", "one_way_edges", "self_loop_edges"]
_UNKNOWN = "unknown"


def export_edge_directionality_summary_csv(
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write directed, reciprocal, one-way, and self-loop counts by edge type."""
    edge_list = list(edges)
    rows = _summary_rows(edge_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "edge_count": len(edge_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _summary_rows(edges: list[KnowledgeEdge | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        groups[_edge_type(edge)].append((_endpoint(edge, "from_unit_id", "source_id", "source"), _endpoint(edge, "to_unit_id", "target_id", "target")))

    rows: list[dict[str, str | int]] = []
    for edge_type in sorted(groups, key=sort_key):
        endpoints = groups[edge_type]
        self_loops = sum(1 for source, target in endpoints if source and source == target)
        directed_counts = defaultdict(int)
        for source, target in endpoints:
            if source and target and source != target:
                directed_counts[(source, target)] += 1
        reciprocal_pairs = 0
        counted: set[frozenset[str]] = set()
        for source, target in directed_counts:
            pair = frozenset((source, target))
            if pair in counted:
                continue
            if directed_counts.get((target, source), 0):
                reciprocal_pairs += 1
                counted.add(pair)
        reciprocal_edge_count = reciprocal_pairs * 2
        non_loop_edges = sum(directed_counts.values())
        rows.append(
            {
                "edge_type": edge_type,
                "total_edges": len(endpoints),
                "reciprocal_pairs": reciprocal_pairs,
                "one_way_edges": max(0, non_loop_edges - reciprocal_edge_count),
                "self_loop_edges": self_loops,
            }
        )
    return rows


def _edge_type(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return field_value(get(edge, "relation")) or field_value(get(edge, "edge_type")) or field_value(get(edge, "type")) or field_value(metadata(edge).get("type")) or _UNKNOWN


def _endpoint(edge: KnowledgeEdge | Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        text = field_value(get(edge, key))
        if text:
            return text
    return ""

