"""Summarize inline HTML kbd tags in Markdown content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import WHITESPACE_RE, field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_KBD_RE = re.compile(r"<kbd\b[^>]*>(.*?)</kbd\s*>", re.IGNORECASE)


def summarize_unit_markdown_kbd_usage(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = count = multi = 0
    keys: Counter[str] = Counter()
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        found = False
        for line_number, key in _keys(str(get(unit, "content") or "")):
            found = True
            count += 1
            keys[key] += 1
            if "+" in key or " " in key:
                multi += 1
            if len(samples) < limit:
                samples.append({"unit_id": unit_id(unit), "line_number": line_number, "key": key})
        if found:
            units_with += 1
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["key"])))
    return {
        "total_units": total,
        "units_with_kbd": units_with,
        "kbd_count": count,
        "key_counts": {key: keys[key] for key in sorted(keys, key=sort_key)},
        "multi_key_sequence_count": multi,
        "samples": samples[:limit],
    }


def _keys(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _KBD_RE.finditer(line):
            key = WHITESPACE_RE.sub(" ", field_value(match.group(1))).strip()
            if key:
                rows.append((line_number, key))
    return rows
