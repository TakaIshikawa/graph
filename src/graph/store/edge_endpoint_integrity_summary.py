"""Summarize relation endpoint integrity."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, sort_key, unit_id

_SOURCE_KEYS = ("source_id", "from_unit_id", "source_unit_id", "from_id", "source")
_TARGET_KEYS = ("target_id", "to_unit_id", "target_unit_id", "to_id", "target")
_RELATION_KEYS = ("relation_type", "relation", "type", "predicate")


def edge_endpoint_integrity_summary(edges: Iterable[Mapping[str, Any] | object], units: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    unit_ids = {unit_id(unit) for unit in units if unit_id(unit)}
    edge_list = list(edges)
    seen: Counter[tuple[str, str, str]] = Counter((_value(edge, _SOURCE_KEYS), _value(edge, _TARGET_KEYS), _value(edge, _RELATION_KEYS) or "unknown") for edge in edge_list)
    rows = []
    counts: Counter[str] = Counter()
    for index, edge in enumerate(edge_list):
        source = _value(edge, _SOURCE_KEYS)
        target = _value(edge, _TARGET_KEYS)
        relation = _value(edge, _RELATION_KEYS) or "unknown"
        status = "ok"
        if not source or source not in unit_ids:
            status = "missing_source"
        if not target or target not in unit_ids:
            status = "missing_target" if status == "ok" else f"{status},missing_target"
        if source and target and source == target:
            status = "self_loop" if status == "ok" else f"{status},self_loop"
        if seen[(source, target, relation)] > 1:
            status = "duplicate_edge" if status == "ok" else f"{status},duplicate_edge"
        for part in status.split(","):
            counts[part] += 1
        rows.append({"edge_id": edge_id(edge) or str(index), "source_id": source, "target_id": target, "relation_type": relation, "status": status})
    rows.sort(key=lambda row: (sort_key(row["status"]), sort_key(row["edge_id"])))
    return {"total_edges": len(edge_list), "status_counts": dict(sorted(counts.items())), "rows": rows}


def _value(item: Any, keys: tuple[str, ...]) -> str:
    meta = metadata(item)
    for key in keys:
        text = field_value(get(item, key)) or field_value(meta.get(key))
        if text:
            return text
    return ""
