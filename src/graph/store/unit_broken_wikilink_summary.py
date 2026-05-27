"""Summarize wikilinks that do not resolve to known units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_TARGET_KEYS = ("id", "unit_id", "source_id", "title", "slug")


def summarize_unit_broken_wikilinks(units: Iterable[Any]) -> dict[str, Any]:
    items = list(units)
    known = {_normalize(value) for unit in items for value in _targets(unit) if _normalize(value)}
    rows = []
    total_links = broken_link_count = 0
    for index, unit in enumerate(items):
        source = unit_id(unit) or str(index)
        missing = set()
        for raw in _WIKILINK_RE.findall(_content(unit)):
            total_links += 1
            target = raw.split("|", 1)[0].strip()
            if _normalize(target) not in known:
                missing.add(target)
        if missing:
            broken_link_count += len(missing)
            rows.append({"unit_id": source, "missing_targets": sorted(missing, key=sort_key), "missing_count": len(missing)})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {"total_units": len(items), "total_wikilinks": total_links, "broken_link_count": broken_link_count, "rows": rows}


def _targets(unit: Any) -> list[str]:
    meta = metadata(unit)
    return [field_value(get(unit, key)) or field_value(meta.get(key)) for key in _TARGET_KEYS]


def _content(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "content")) or field_value(meta.get("content"))


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()
