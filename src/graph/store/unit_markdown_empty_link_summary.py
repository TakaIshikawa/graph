"""Summarize Markdown inline links with empty destinations."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)]*)\)")


def summarize_unit_markdown_empty_links(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = empty_count = anchor_count = 0
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        links = _links(str(get(unit, "content") or ""))
        if links:
            units_with += 1
        for line_number, label, destination in links:
            empty_count += 1
            anchor_count += 1 if destination.strip() == "#" else 0
            if len(samples) < limit:
                samples.append({"unit_id": unit_id(unit), "line_number": line_number, "label": label, "destination": field_value(destination)})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    return {"total_units": total, "units_with_empty_links": units_with, "empty_link_count": empty_count, "anchor_placeholder_count": anchor_count, "samples": samples[:limit]}


def _links(content: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _LINK_RE.finditer(line):
            destination = match.group(2)
            if destination.strip() in {"", "#"}:
                rows.append((line_number, field_value(match.group(1)), destination))
    return rows
