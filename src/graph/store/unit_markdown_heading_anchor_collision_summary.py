"""Summarize duplicate generated Markdown heading anchors within units."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def summarize_unit_markdown_heading_anchor_collisions(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    unit_list = list(units)
    rows: list[dict[str, Any]] = []
    duplicate_anchor_count = 0
    for unit in unit_list:
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for heading in _headings(str(get(unit, "content") or "")):
            grouped[heading["anchor"]].append(heading)
        collisions = []
        for anchor, headings in grouped.items():
            if len(headings) > 1:
                duplicate_anchor_count += 1
                collisions.append(
                    {
                        "anchor": anchor,
                        "heading_count": len(headings),
                        "levels": sorted({heading["level"] for heading in headings}),
                        "sample_headings": [heading["text"] for heading in headings[:limit]],
                    }
                )
        collisions.sort(key=lambda row: sort_key(row["anchor"]))
        if collisions:
            rows.append({"unit_id": unit_id(unit), "duplicate_anchor_count": len(collisions), "collisions": collisions})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {"total_units": len(unit_list), "affected_unit_count": len(rows), "duplicate_anchor_count": duplicate_anchor_count, "units": rows}


def _headings(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if match := _HEADING_RE.match(line):
            text = re.sub(r"\s+#+\s*$", "", match.group(2)).strip()
            rows.append({"level": len(match.group(1)), "text": text, "anchor": _anchor(text)})
    return rows


def _anchor(text: str) -> str:
    value = field_value(text).casefold()
    value = re.sub(r"[^\w\s-]", "", value)
    return re.sub(r"[\s_]+", "-", value).strip("-")
