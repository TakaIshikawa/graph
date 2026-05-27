"""Summarize Markdown backslash escapes by source."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ESCAPE_RE = re.compile(r"\\([\\`*{}\[\]()#+\-.!_>~|=])")


def summarize_unit_markdown_escapes(units: Iterable[Any]) -> dict[str, Any]:
    total_units = 0
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_count": 0, "units_with": 0, "escapes": 0, "dangling": 0, "chars": Counter()})
    for unit in units:
        total_units += 1
        text = _content_without_fences(unit)
        chars = _ESCAPE_RE.findall(text)
        dangling = sum(1 for line in text.splitlines() if line.endswith("\\") and not line.endswith("\\\\"))
        group = groups[_source(unit)]
        group["unit_count"] += 1
        group["escapes"] += len(chars)
        group["dangling"] += dangling
        group["chars"].update(chars)
        if chars or dangling:
            group["units_with"] += 1
    rows = []
    for source, data in groups.items():
        chars: Counter[str] = data["chars"]
        common = min((char for char, count in chars.items() if count == max(chars.values(), default=0)), key=sort_key, default="")
        rows.append({
            "source": source,
            "unit_count": data["unit_count"],
            "units_with_escapes": data["units_with"],
            "escape_sequence_count": data["escapes"],
            "dangling_backslash_count": data["dangling"],
            "most_common_escaped_character": common,
        })
    rows.sort(key=lambda row: sort_key(row["source"]))
    return {"total_units": total_units, "sources": rows}


def _content_without_fences(unit: Any) -> str:
    lines: list[str] = []
    in_fence = False
    for line in str(get(unit, "content") or "").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def _source(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "source") or get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"
