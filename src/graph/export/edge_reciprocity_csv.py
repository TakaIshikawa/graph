"""CSV export for directed edge reciprocity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, write_csv

_FIELDNAMES = ["source_id", "target_id", "relation", "has_reverse", "reverse_relations", "reciprocity_group"]


def export_edge_reciprocity_csv(
    edges: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per directed edge with reverse-edge details."""
    edge_list = list(edges)
    rows = _rows(edge_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "edge_count": len(edge_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(edges: list[Mapping[str, Any] | object]) -> list[dict[str, str]]:
    by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for edge in edges:
        by_pair[(_source(edge), _target(edge))].add(_relation(edge))
    rows = []
    for edge in edges:
        source = _source(edge)
        target = _target(edge)
        relation = _relation(edge)
        reverse_relations = sorted(by_pair.get((target, source), set()), key=sort_key)
        is_self_loop = source == target
        has_reverse = bool(reverse_relations) and not is_self_loop
        rows.append(
            {
                "source_id": source,
                "target_id": target,
                "relation": relation,
                "has_reverse": "true" if has_reverse else "false",
                "reverse_relations": "; ".join(reverse_relations if not is_self_loop else []),
                "reciprocity_group": "self_loop" if is_self_loop else ("reciprocal" if has_reverse else "one_way"),
            }
        )
    return sorted(rows, key=lambda row: (sort_key(row["source_id"]), sort_key(row["target_id"]), sort_key(row["relation"])))


def _source(edge: Mapping[str, Any] | object) -> str:
    return field_value(get(edge, "source_id") or get(edge, "from_unit_id") or get(edge, "source_unit_id"))


def _target(edge: Mapping[str, Any] | object) -> str:
    return field_value(get(edge, "target_id") or get(edge, "to_unit_id") or get(edge, "target_unit_id"))


def _relation(edge: Mapping[str, Any] | object) -> str:
    return field_value(get(edge, "relation") or get(edge, "relation_type")) or "related"
