"""Summarize Markdown list marker styles."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_MARKER_RE = re.compile(r"^\s{0,3}([*+-]|[0-9]+[.)]|[ivxlcdmIVXLCDM]+[.)])\s+")


def summarize_unit_markdown_list_marker_styles(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize unordered, numeric ordered, and roman ordered list markers."""
    limit = max(0, sample_limit)
    total = units_with_lists = mixed = 0
    counts: Counter[str] = Counter()
    examples: list[dict[str, str | int | dict[str, int]]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        unit_counts: Counter[str] = Counter(style for _, style in _markers(str(get(unit, "content") or "")))
        if not unit_counts:
            continue
        units_with_lists += 1
        counts.update(unit_counts)
        if len(unit_counts) > 1:
            mixed += 1
            examples.append({"unit_id": uid, "marker_counts": {key: unit_counts[key] for key in sorted(unit_counts, key=sort_key)}})
    examples.sort(key=lambda row: sort_key(row["unit_id"]))
    return {
        "total_units": total,
        "units_with_lists": units_with_lists,
        "marker_counts": {key: counts[key] for key in sorted(counts, key=sort_key)},
        "units_with_mixed_markers": mixed,
        "examples": examples[:limit],
    }


def _markers(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _MARKER_RE.match(line)
        if match:
            rows.append((line_number, _style(match.group(1))))
    return rows


def _style(marker: str) -> str:
    if marker == "-":
        return "unordered_dash"
    if marker == "*":
        return "unordered_star"
    if marker == "+":
        return "unordered_plus"
    return "ordered_roman" if marker[:-1].isalpha() else "ordered_numeric"
