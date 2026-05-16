"""CSV export for per-unit relation degree counts."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "in_degree",
    "out_degree",
    "total_degree",
    "distinct_neighbor_count",
    "neighbor_ids",
]
_WHITESPACE_RE = re.compile(r"\s+")
_SOURCE_KEYS = ("from_unit_id", "source_unit_id", "source_id", "from_id", "source", "from")
_TARGET_KEYS = ("to_unit_id", "target_unit_id", "target_id", "to_id", "target", "to")


def export_unit_relation_degree_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write deterministic per-unit relation degree counts."""
    unit_list = list(units)
    edge_list = list(edges)
    rows = _degree_rows(unit_list, edge_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "edge_count": len(edge_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _degree_rows(
    units: list[KnowledgeUnit | Mapping[str, Any]],
    edges: list[KnowledgeEdge | Mapping[str, Any]],
) -> list[dict[str, str | int]]:
    unit_ids = {_unit_id(unit) for unit in units if _unit_id(unit)}
    in_degrees = {unit_id: 0 for unit_id in unit_ids}
    out_degrees = {unit_id: 0 for unit_id in unit_ids}
    neighbors = {unit_id: set() for unit_id in unit_ids}

    for edge in edges:
        source_id = _edge_endpoint(edge, _SOURCE_KEYS)
        target_id = _edge_endpoint(edge, _TARGET_KEYS)
        if not source_id or not target_id:
            continue

        if source_id in unit_ids:
            out_degrees[source_id] += 1
            if target_id:
                neighbors[source_id].add(target_id)
        if target_id in unit_ids:
            in_degrees[target_id] += 1
            if source_id:
                neighbors[target_id].add(source_id)

    rows: list[dict[str, str | int]] = []
    for unit in units:
        unit_id = _unit_id(unit)
        in_degree = in_degrees.get(unit_id, 0)
        out_degree = out_degrees.get(unit_id, 0)
        unit_neighbors = neighbors.get(unit_id, set())
        rows.append(
            {
                "unit_id": unit_id,
                "title": _field_value(_get(unit, "title")),
                "source_project": _field_value(_get(unit, "source_project")) or "Unknown",
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree,
                "distinct_neighbor_count": len(unit_neighbors),
                "neighbor_ids": "; ".join(sorted(unit_neighbors, key=_sort_key)),
            }
        )

    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["title"])))


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _edge_endpoint(edge: KnowledgeEdge | Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _endpoint_text(_get(edge, key))
        if text:
            return text
    return ""


def _endpoint_text(value: object) -> str:
    if isinstance(value, Mapping):
        return _field_value(value.get("id")) or _field_value(value.get("unit_id"))
    object_id = _field_value(getattr(value, "id", None)) or _field_value(getattr(value, "unit_id", None))
    if object_id:
        return object_id
    return _field_value(value)


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
