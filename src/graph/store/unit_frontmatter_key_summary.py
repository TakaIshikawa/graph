"""Summarize keys declared in leading markdown frontmatter blocks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_KEY_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*:", re.MULTILINE)


def summarize_unit_frontmatter_keys(units: Iterable[Any]) -> dict[str, Any]:
    total_units = units_with_frontmatter = units_missing_frontmatter = 0
    key_counts: Counter[str] = Counter()
    duplicate_key_units: list[dict[str, Any]] = []

    for index, unit in enumerate(units):
        total_units += 1
        frontmatter = _frontmatter(_content(unit))
        if frontmatter is None:
            units_missing_frontmatter += 1
            continue
        units_with_frontmatter += 1
        keys = [_normalize_key(match.group(1)) for match in _KEY_RE.finditer(frontmatter)]
        key_counts.update(dict.fromkeys(keys, 1))
        duplicates = sorted([key for key, count in Counter(keys).items() if count > 1], key=sort_key)
        if duplicates:
            duplicate_key_units.append({"unit_id": unit_id(unit) or str(index), "duplicate_keys": duplicates})

    duplicate_key_units.sort(key=lambda row: sort_key(row["unit_id"]))
    return {
        "total_units": total_units,
        "units_with_frontmatter": units_with_frontmatter,
        "units_missing_frontmatter": units_missing_frontmatter,
        "key_counts": [{"key": key, "count": key_counts[key]} for key in sorted(key_counts, key=sort_key)],
        "duplicate_key_units": duplicate_key_units,
    }


def _frontmatter(content: str) -> str | None:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index])
    return None


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)


def _normalize_key(key: str) -> str:
    return key.strip().casefold()
