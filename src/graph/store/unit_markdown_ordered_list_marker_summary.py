"""Summarize Markdown ordered-list markers in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ITEM_RE = re.compile(r"^\s{0,3}(\d+)([.)])\s+(.*)$")


def summarize_unit_markdown_ordered_list_markers(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = items = non_one = paren = dot = 0
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        markers = _markers(str(get(unit, "content") or ""))
        if markers:
            units_with += 1
        for line_number, number, delimiter, text in markers:
            items += 1
            non_one += 1 if number != 1 else 0
            paren += 1 if delimiter == ")" else 0
            dot += 1 if delimiter == "." else 0
            if len(samples) < limit:
                samples.append({"unit_id": unit_id(unit), "line_number": line_number, "marker_number": number, "delimiter": delimiter, "item_text": text})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    return {"total_units": total, "units_with_ordered_lists": units_with, "item_count": items, "non_one_start_count": non_one, "paren_delimiter_count": paren, "dot_delimiter_count": dot, "samples": samples[:limit]}


def _markers(content: str) -> list[tuple[int, int, str, str]]:
    rows: list[tuple[int, int, str, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _ITEM_RE.match(line)
        if match:
            rows.append((line_number, int(match.group(1)), match.group(2), field_value(match.group(3))))
    return rows
