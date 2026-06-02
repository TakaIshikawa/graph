"""CSV export for relations with missing endpoints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["relation_id", "source_id", "target_id", "missing_source", "missing_target", "relation_type"]
_SOURCE_KEYS = ("source_id", "from_unit_id", "source_unit_id", "from_id", "source")
_TARGET_KEYS = ("target_id", "to_unit_id", "target_unit_id", "to_id", "target")
_RELATION_KEYS = ("relation_type", "relation", "type", "predicate")


def export_relation_endpoint_orphan_csv(
    units: Iterable[Mapping[str, Any] | object],
    edges: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_ids = {unit_id(unit) for unit in units if unit_id(unit)}
    edge_list = list(edges)
    rows = []
    for index, edge in enumerate(edge_list):
        source = _value(edge, _SOURCE_KEYS)
        target = _value(edge, _TARGET_KEYS)
        missing_source = not source or source not in unit_ids
        missing_target = not target or target not in unit_ids
        if not missing_source and not missing_target:
            continue
        rows.append(
            {
                "relation_id": edge_id(edge) or str(index),
                "source_id": source,
                "target_id": target,
                "missing_source": _bool_text(missing_source),
                "missing_target": _bool_text(missing_target),
                "relation_type": _value(edge, _RELATION_KEYS) or "unknown",
            }
        )
    rows.sort(key=lambda row: (sort_key(row["relation_type"]), sort_key(row["relation_id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "relation_count": len(edge_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _value(item: Any, keys: tuple[str, ...]) -> str:
    meta = metadata(item)
    for key in keys:
        text = field_value(get(item, key)) or field_value(meta.get(key))
        if text:
            return text
    return ""


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
