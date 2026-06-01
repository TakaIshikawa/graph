"""Summarize Obsidian-style highlight marks in Markdown units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HIGHLIGHT_RE = re.compile(r"(?<!\\)==(.+?)(?<!\\)==")


def summarize_unit_markdown_highlight_marks(units: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    """Summarize closed ``==highlight==`` spans per unit."""
    rows: list[dict[str, Any]] = []
    total_units = total_highlights = 0
    for index, unit in enumerate(units):
        total_units += 1
        uid = unit_id(unit) or str(index)
        highlights = [field_value(match.group(1)) for line in _lines(_content(unit)) for match in _HIGHLIGHT_RE.finditer(line)]
        lengths = [len(text) for text in highlights]
        total_highlights += len(highlights)
        rows.append({
            "unit_id": uid,
            "highlight_count": len(highlights),
            "first_highlight": highlights[0] if highlights else "",
            "min_highlight_length": min(lengths) if lengths else 0,
            "max_highlight_length": max(lengths) if lengths else 0,
            "average_highlight_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        })
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {"total_units": total_units, "total_highlights": total_highlights, "units": rows}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _lines(content: str) -> list[str]:
    rows: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append(line)
    return rows
