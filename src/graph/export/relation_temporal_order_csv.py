"""CSV export for temporal ordering of relation endpoints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, parse_datetime, render_csv, sort_key, write_csv

_FIELDNAMES = ["relation_id", "source_id", "target_id", "source_timestamp", "target_timestamp", "order", "lag_days"]
_SOURCE_KEYS = ("source_timestamp", "source_time", "source_date")
_TARGET_KEYS = ("target_timestamp", "target_time", "target_date")


def export_relation_temporal_order_csv(relations: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    relation_list = list(relations)
    rows = [_row(relation) for relation in relation_list]
    rows.sort(key=lambda row: (sort_key(row["relation_id"]), sort_key(row["source_id"]), sort_key(row["target_id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "relation_count": len(relation_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(relation: Mapping[str, Any] | object) -> dict[str, str]:
    source_raw = _first(relation, _SOURCE_KEYS)
    target_raw = _first(relation, _TARGET_KEYS)
    source_dt = parse_datetime(source_raw)
    target_dt = parse_datetime(target_raw)
    order = "unknown"
    lag_days = ""
    if source_dt and target_dt:
        delta = target_dt - source_dt
        lag_days = f"{delta.total_seconds() / 86400:.6g}"
        if delta.total_seconds() > 0:
            order = "source_before_target"
        elif delta.total_seconds() < 0:
            order = "target_before_source"
        else:
            order = "same_time"
    return {
        "relation_id": edge_id(relation),
        "source_id": field_value(get(relation, "source_id") or get(relation, "source")),
        "target_id": field_value(get(relation, "target_id") or get(relation, "target")),
        "source_timestamp": field_value(source_raw),
        "target_timestamp": field_value(target_raw),
        "order": order,
        "lag_days": lag_days,
    }


def _first(relation: Mapping[str, Any] | object, keys: tuple[str, ...]) -> object:
    meta = metadata(relation)
    for key in keys:
        value = get(relation, key)
        if field_value(value):
            return value
        value = meta.get(key)
        if field_value(value):
            return value
    return ""
