"""Summarize leading YAML frontmatter presence in unit content."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_KEY_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*:", re.MULTILINE)


def summarize_unit_frontmatter_presence(units: Iterable[Any]) -> dict[str, Any]:
    counts = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_entity_type: dict[str, Counter[str]] = defaultdict(Counter)
    key_counts: Counter[str] = Counter()
    total_units = 0

    for unit in units:
        total_units += 1
        status, block = _status(_content(unit))
        counts[status] += 1
        by_source[_source(unit)][status] += 1
        by_entity_type[_entity_type(unit)][status] += 1
        if status == "valid":
            key_counts.update(dict.fromkeys(_keys(block), 1))

    return {
        "total_units": total_units,
        "valid_frontmatter_units": counts["valid"],
        "empty_frontmatter_units": counts["empty"],
        "malformed_frontmatter_units": counts["malformed"],
        "missing_frontmatter_units": counts["missing"],
        "by_source": _group_rows(by_source),
        "by_entity_type": _group_rows(by_entity_type),
        "top_frontmatter_keys": [{"key": key, "count": key_counts[key]} for key in sorted(key_counts, key=sort_key)],
    }


def _status(content: str) -> tuple[str, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return "missing", ""
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            block = "\n".join(lines[1:index]).strip()
            if not block:
                return "empty", ""
            return ("valid" if _keys(block) else "malformed"), block
    return "malformed", ""


def _keys(block: str) -> list[str]:
    return [_normalize_key(match.group(1)) for match in _KEY_RE.finditer(block)]


def _group_rows(groups: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(groups, key=sort_key):
        counter = groups[name]
        rows.append(
            {
                "name": name,
                "valid": counter["valid"],
                "empty": counter["empty"],
                "malformed": counter["malformed"],
                "missing": counter["missing"],
            }
        )
    return rows


def _source(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "source") or get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"


def _entity_type(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "entity_type") or meta.get("entity_type") or meta.get("type")) or "unknown"


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)


def _normalize_key(key: str) -> str:
    return key.strip().casefold()
