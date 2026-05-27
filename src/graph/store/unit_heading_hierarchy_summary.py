"""Summarize Markdown heading hierarchy health across units."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def summarize_unit_heading_hierarchy(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total_units = units_with_headings = units_with_skips = max_depth = 0
    counts: Counter[int] = Counter()
    samples: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        total_units += 1
        headings = _headings(unit)
        if headings:
            units_with_headings += 1
        unit_skips = [heading for heading in headings if heading["skipped_level"]]
        if unit_skips:
            units_with_skips += 1
            if len(samples) < limit:
                samples.append({"unit_id": unit_id(unit) or str(index), "title": _title(unit), "line_number": unit_skips[0]["line_number"], "level": unit_skips[0]["level"]})
        for heading in headings:
            counts[heading["level"]] += 1
            max_depth = max(max_depth, heading["level"])
    return {
        "total_units": total_units,
        "units_with_headings": units_with_headings,
        "heading_counts_by_level": {str(level): counts[level] for level in sorted(counts)},
        "max_depth": max_depth,
        "units_with_skipped_levels": units_with_skips,
        "skipped_level_samples": sorted(samples, key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]))),
    }


def _headings(unit: Any) -> list[dict[str, Any]]:
    stack: list[int] = []
    rows: list[dict[str, Any]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        while stack and stack[-1] >= level:
            stack.pop()
        expected = stack[-1] + 1 if stack else 1
        rows.append({"level": level, "line_number": line_number, "skipped_level": level > expected})
        stack.append(level)
    return rows


def _title(unit: Any) -> str:
    return field_value(get(unit, "title") or metadata(unit).get("title"))
