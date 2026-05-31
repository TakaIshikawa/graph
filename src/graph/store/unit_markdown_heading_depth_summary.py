"""Summarize Markdown ATX heading depths in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")


def summarize_unit_markdown_heading_depths(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = count = skipped = 0
    depth_counts = {str(depth): 0 for depth in range(1, 7)}
    max_depth = 0
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        headings = _headings(str(get(unit, "content") or ""))
        if headings:
            units_with += 1
        previous = 0
        for line_number, depth, text in headings:
            count += 1
            depth_counts[str(depth)] += 1
            max_depth = max(max_depth, depth)
            if previous and depth > previous + 1:
                skipped += 1
            previous = depth
            if len(samples) < limit:
                samples.append({"unit_id": unit_id(unit), "line_number": line_number, "depth": depth, "heading_text": text})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    return {"total_units": total, "units_with_headings": units_with, "heading_count": count, "depth_counts": depth_counts, "max_depth": max_depth, "skipped_level_count": skipped, "samples": samples[:limit]}


def _headings(content: str) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            rows.append((line_number, len(match.group(1)), field_value(match.group(2))))
    return rows
