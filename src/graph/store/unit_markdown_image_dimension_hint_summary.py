"""Summarize Markdown image dimension hints."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_WIKI_RE = re.compile(r"!\[\[([^\]|]+)\|([0-9]+)(?:x([0-9]+))?]]")
_MD_RE = re.compile(r"!\[[^\]\n]*]\(([^)\n]*?)\s+=([0-9]+)(?:x([0-9]+))?\)")


def summarize_unit_markdown_image_dimension_hints(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize Obsidian and Markdown image dimension hints."""
    limit = max(0, sample_limit)
    total = count = 0
    affected: set[str] = set()
    hint_counts: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        for line_number, target, width, height in _hints(str(get(unit, "content") or "")):
            count += 1
            affected.add(uid)
            hint_type = "width_height" if height else "width_only"
            hint_counts[hint_type] += 1
            examples.append({"unit_id": uid, "line": line_number, "target": target, "width": width, "height": height})
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["target"])))
    return {
        "total_units": total,
        "dimension_hint_count": count,
        "affected_units": len(affected),
        "hint_counts": {key: hint_counts[key] for key in sorted(hint_counts, key=sort_key)},
        "examples": examples[:limit],
    }


def _hints(content: str) -> list[tuple[int, str, str, str]]:
    rows: list[tuple[int, str, str, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _WIKI_RE.finditer(line):
            rows.append((line_number, field_value(match.group(1)), match.group(2), match.group(3) or ""))
        for match in _MD_RE.finditer(line):
            rows.append((line_number, field_value(match.group(1)), match.group(2), match.group(3) or ""))
    return rows
