"""Duplicate source path summary for store units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def summarize_unit_duplicate_source_paths(units: Iterable[Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"source_path": "", "unit_ids": [], "source_types": set()})
    total = skipped = 0
    for index, unit in enumerate(units):
        total += 1
        path = _source_path(unit)
        if not path:
            skipped += 1
            continue
        normalized = _normalize(path)
        group = groups[normalized]
        group["source_path"] = normalized
        group["unit_ids"].append(_unit_id(unit, index))
        source_type = _source_type(unit)
        if source_type:
            group["source_types"].add(source_type)
    duplicates = []
    for path in sorted(groups, key=_sort_key):
        group = groups[path]
        if len(group["unit_ids"]) < 2:
            continue
        duplicates.append(
            {
                "source_path": path,
                "unit_ids": sorted(group["unit_ids"], key=_sort_key),
                "count": len(group["unit_ids"]),
                "source_types": sorted(group["source_types"], key=_sort_key),
            }
        )
    return {"total_units": total, "skipped_units": skipped, "duplicate_paths": duplicates}


def _source_path(unit: Any) -> str:
    meta = _metadata(unit)
    return _text(_get(unit, "source_path")) or _text(meta.get("source_path"))


def _source_type(unit: Any) -> str:
    meta = _metadata(unit)
    return _text(_get(unit, "source_type")) or _text(_get(unit, "source_project")) or _text(meta.get("source_type")) or _text(meta.get("source_project"))


def _normalize(path: str) -> str:
    path = path.strip().replace("\\", "/")
    while "//" in path and "://" not in path:
        path = path.replace("//", "/")
    return path.rstrip("/") or path


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _unit_id(unit: Any, index: int) -> str:
    return _text(_get(unit, "id") or _metadata(unit).get("id")) or str(index)


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(getattr(value, "value", value)).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
