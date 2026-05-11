"""Adjacency-list CSV export helpers."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = ["source_id", "target_id", "edge_type", "edge_label", "source_label", "target_label"]


def export_graph_adjacency_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path,
) -> dict[str, Any]:
    """Write one adjacency-list CSV row per edge, preserving edge input order."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unit_list = list(units)
    edge_list = list(edges)
    labels = {_unit_id(unit): _text(unit.title) for unit in unit_list}

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for edge in edge_list:
            writer.writerow(_edge_row(edge, labels))

    return {
        "path": str(output_path),
        "nodes_scanned": len(unit_list),
        "edges_exported": len(edge_list),
        "bytes_written": output_path.stat().st_size,
    }


def _edge_row(edge: KnowledgeEdge, labels: dict[str, str]) -> dict[str, str]:
    source_id = _text(edge.from_unit_id)
    target_id = _text(edge.to_unit_id)
    relation = _enum_value(edge.relation)
    return {
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": relation,
        "edge_label": _text(edge.metadata.get("label") or edge.metadata.get("title") or relation),
        "source_label": labels.get(source_id, ""),
        "target_label": labels.get(target_id, ""),
    }


def _unit_id(unit: KnowledgeUnit) -> str:
    return _text(unit.id or unit.source_id)


def _enum_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _text(value: object) -> str:
    return str(value or "")
