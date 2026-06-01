"""Summarize resolved Markdown reference-style link usage."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEF_RE = re.compile(r"^[ \t]{0,3}\[([^\]\n]+)]\s*:")
_FULL_RE = re.compile(r"(?<!!)\[([^\]\n]+)]\[([^\]\n]+)]")
_COLLAPSED_RE = re.compile(r"(?<!!)\[([^\]\n]+)]\[\]")
_IMAGE_REF_RE = re.compile(r"!\[[^\]\n]*]\[[^\]\n]*]")
_BRACKET_RE = re.compile(r"(?<!!)\[([^\]\n]+)](?!\(|\[|:)")


def summarize_unit_markdown_reference_link_usage(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize full, collapsed, and shortcut references resolved in each unit."""
    limit = max(0, sample_limit)
    total = units_with = 0
    counts: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        lines = _content_lines(str(get(unit, "content") or ""))
        definitions = {_normalize(match.group(1)) for _, line in lines if (match := _DEF_RE.match(line))}
        unit_count = 0
        for line_number, line in lines:
            if _DEF_RE.match(line):
                continue
            occupied: list[range] = [range(match.start(), match.end()) for match in _IMAGE_REF_RE.finditer(line)]
            for match in _FULL_RE.finditer(line):
                occupied.append(range(match.start(), match.end()))
                if _normalize(match.group(2)) in definitions:
                    unit_count += 1
                    counts["full"] += 1
                    examples.append({"unit_id": uid, "line": line_number, "usage_type": "full", "label": field_value(match.group(2))})
            for match in _COLLAPSED_RE.finditer(line):
                occupied.append(range(match.start(), match.end()))
                if _normalize(match.group(1)) in definitions:
                    unit_count += 1
                    counts["collapsed"] += 1
                    examples.append({"unit_id": uid, "line": line_number, "usage_type": "collapsed", "label": field_value(match.group(1))})
            for match in _BRACKET_RE.finditer(line):
                if any(match.start() in item for item in occupied):
                    continue
                if _normalize(match.group(1)) in definitions:
                    unit_count += 1
                    counts["shortcut"] += 1
                    examples.append({"unit_id": uid, "line": line_number, "usage_type": "shortcut", "label": field_value(match.group(1))})
        if unit_count:
            units_with += 1
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["usage_type"]), sort_key(row["label"])))
    return {
        "total_units": total,
        "units_with_reference_link_usage": units_with,
        "full_reference_count": counts["full"],
        "collapsed_reference_count": counts["collapsed"],
        "shortcut_reference_count": counts["shortcut"],
        "examples": examples[:limit],
    }


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", field_value(value)).casefold()
