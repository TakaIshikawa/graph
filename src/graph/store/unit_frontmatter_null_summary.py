"""Summarize null-like YAML frontmatter values."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get

_FIELD_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.*?)\s*(?:#.*)?$")


def summarize_unit_frontmatter_nulls(units: Iterable[Any]) -> dict[str, Any]:
    total = units_with = 0
    key_counts: Counter[str] = Counter()
    null_kind_counts: Counter[str] = Counter()
    for unit in units:
        found = False
        for key, kind in _nulls(str(get(unit, "content") or "")):
            total += 1; found = True; key_counts[key] += 1; null_kind_counts[kind] += 1
        if found:
            units_with += 1
    return {"total_null_values": total, "units_with_null_values": units_with, "key_counts": dict(sorted(key_counts.items())), "null_kind_counts": dict(sorted(null_kind_counts.items()))}


def _nulls(content: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return rows
    for line in lines[1:]:
        if line.strip() == "---":
            return rows
        if match := _FIELD_RE.match(line):
            value = match.group(2).strip()
            lowered = value.casefold()
            if value == "":
                rows.append((match.group(1), "blank"))
            elif lowered in {"null", "~"}:
                rows.append((match.group(1), "tilde" if lowered == "~" else "null"))
    return []
