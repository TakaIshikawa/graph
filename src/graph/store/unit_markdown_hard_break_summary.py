"""Summarize Markdown hard line breaks by source."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def summarize_unit_markdown_hard_breaks(units: Iterable[Any]) -> dict[str, Any]:
    total_units = 0
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_count": 0, "units_with": 0, "hard": 0, "spaces": 0, "backslash": 0, "max": 0})
    for unit in units:
        total_units += 1
        spaces, backslash = _counts(unit)
        hard = spaces + backslash
        group = groups[_source(unit)]
        group["unit_count"] += 1
        group["hard"] += hard
        group["spaces"] += spaces
        group["backslash"] += backslash
        group["max"] = max(group["max"], hard)
        if hard:
            group["units_with"] += 1
    rows = [
        {
            "source": source,
            "unit_count": data["unit_count"],
            "units_with_hard_breaks": data["units_with"],
            "hard_break_count": data["hard"],
            "trailing_space_break_count": data["spaces"],
            "backslash_break_count": data["backslash"],
            "max_hard_breaks_per_unit": data["max"],
        }
        for source, data in groups.items()
    ]
    rows.sort(key=lambda row: sort_key(row["source"]))
    return {"total_units": total_units, "sources": rows}


def _counts(unit: Any) -> tuple[int, int]:
    spaces = 0
    backslash = 0
    in_fence = False
    lines = str(get(unit, "content") or "").splitlines()
    for line in lines[:-1]:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.endswith("\\") and not line.endswith("\\\\"):
            backslash += 1
        elif len(line) - len(line.rstrip(" ")) >= 2:
            spaces += 1
    return spaces, backslash


def _source(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "source") or get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"
