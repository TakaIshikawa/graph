"""Summarize repeated prefixes in unit titles."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_BRACKET_RE = re.compile(r"^\[([^\]]+)\]")
_STATUS_RE = re.compile(r"^(TODO|DONE|FIXME|NOTE)\b", re.IGNORECASE)


def summarize_unit_title_prefixes(units: Iterable[Any], min_count: int = 2) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_ids": [], "sources": defaultdict(int)})
    total_units = 0
    for index, unit in enumerate(units):
        total_units += 1
        meta = metadata(unit)
        prefix = _prefix(_first(unit, meta, ("title", "name")))
        if not prefix:
            continue
        group = groups[prefix.casefold()]
        group["unit_ids"].append(unit_id(unit) or str(index))
        group["sources"][_first(unit, meta, ("source", "source_id", "entity_type", "type")) or "unknown"] += 1
    rows = []
    for prefix, group in groups.items():
        if len(group["unit_ids"]) < min_count:
            continue
        rows.append(
            {
                "prefix": prefix,
                "count": len(group["unit_ids"]),
                "example_unit_ids": sorted(group["unit_ids"], key=sort_key)[:5],
                "source_counts": [{"source": source, "count": group["sources"][source]} for source in sorted(group["sources"], key=sort_key)],
            }
        )
    rows.sort(key=lambda row: (-row["count"], sort_key(row["prefix"])))
    return {"total_units": total_units, "prefix_counts": rows}


def _prefix(title: str) -> str:
    title = title.strip()
    if not title:
        return ""
    if match := _ISO_DATE_RE.match(title):
        return match.group(0)
    if match := _BRACKET_RE.match(title):
        return f"[{match.group(1).strip()}]"
    if match := _STATUS_RE.match(title):
        return match.group(1)
    for delimiter in (":", "/"):
        if delimiter in title:
            prefix = title.split(delimiter, 1)[0].strip()
            if prefix:
                return prefix
    return ""


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
