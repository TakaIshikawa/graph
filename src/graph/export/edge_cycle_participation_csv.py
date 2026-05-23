"""CSV export for directed edge cycle participation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import networkx as nx

from graph.export._report_csv import field_value, get, render_csv, sort_key, write_csv

_FIELDNAMES = ["source_id", "target_id", "relation", "cycle_count", "shortest_cycle_length"]


def export_edge_cycle_participation_csv(
    edges: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    *,
    include_zero: bool = True,
) -> str | dict[str, Any]:
    """Return or write directed cycle participation metrics for edges."""
    edge_list = list(edges)
    rows = _rows(edge_list, include_zero)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {
        "path": output_path,
        "edge_count": len(edge_list),
        "rows_exported": len(rows),
        "bytes_written": bytes_written,
    }


def _rows(
    edges: list[Mapping[str, Any] | object], include_zero: bool
) -> list[dict[str, str | int]]:
    graph = nx.DiGraph()
    edge_rows = []
    for edge in edges:
        source = field_value(
            get(edge, "from_unit_id") or get(edge, "source_id") or get(edge, "from_id")
        )
        target = field_value(
            get(edge, "to_unit_id") or get(edge, "target_id") or get(edge, "to_id")
        )
        relation = field_value(get(edge, "relation"))
        if source and target:
            graph.add_edge(source, target)
        edge_rows.append((source, target, relation))

    counts: dict[tuple[str, str], int] = defaultdict(int)
    shortest: dict[tuple[str, str], int] = {}
    for cycle in nx.simple_cycles(graph):
        length = len(cycle)
        for index, source in enumerate(cycle):
            pair = (source, cycle[(index + 1) % length])
            counts[pair] += 1
            shortest[pair] = min(shortest.get(pair, length), length)

    rows = [
        {
            "source_id": source,
            "target_id": target,
            "relation": relation,
            "cycle_count": counts[(source, target)],
            "shortest_cycle_length": shortest.get((source, target), ""),
        }
        for source, target, relation in edge_rows
        if include_zero or counts[(source, target)] > 0
    ]
    return sorted(
        rows,
        key=lambda row: (
            sort_key(row["source_id"]),
            sort_key(row["target_id"]),
            sort_key(row["relation"]),
        ),
    )
