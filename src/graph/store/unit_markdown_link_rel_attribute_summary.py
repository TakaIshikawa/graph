"""Summarize rel attributes on raw HTML anchor tags in Markdown."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ANCHOR_RE = re.compile(r"<a\b([^>]*)>", re.IGNORECASE)
_REL_RE = re.compile(r"""\brel\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)


def summarize_unit_markdown_link_rel_attributes(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize rel attribute values in raw HTML anchor tags."""
    limit = max(0, sample_limit)
    total = anchors = missing = 0
    affected: set[str] = set()
    values: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        for line_number, attrs in _anchors(str(get(unit, "content") or "")):
            anchors += 1
            affected.add(uid)
            match = _REL_RE.search(attrs)
            if not match:
                missing += 1
                examples.append({"unit_id": uid, "line": line_number, "rel": ""})
                continue
            rel_values = [value.casefold() for value in field_value(match.group(1) or match.group(2) or match.group(3)).split() if value]
            for value in rel_values:
                values[value] += 1
                examples.append({"unit_id": uid, "line": line_number, "rel": value})
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["rel"])))
    return {
        "total_units": total,
        "anchor_count": anchors,
        "missing_rel_anchor_count": missing,
        "affected_units": len(affected),
        "rel_value_counts": {key: values[key] for key in sorted(values, key=sort_key)},
        "examples": examples[:limit],
    }


def _anchors(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _ANCHOR_RE.finditer(line):
            rows.append((line_number, match.group(1)))
    return rows
