"""Summarize Markdown footnote definitions."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_DEF_RE = re.compile(r"^\[\^([^\]\n]+)\]:\s*(.*)$")


def summarize_unit_markdown_footnote_definitions(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = multiline = 0
    units_with: set[str] = set()
    label_counts: Counter[str] = Counter()
    duplicate_labels: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for unit in units:
        uid = unit_id(unit)
        seen: Counter[str] = Counter()
        for label, definition, line, continued in _definitions(str(get(unit, "content") or "")):
            total += 1
            units_with.add(uid)
            label_counts[label] += 1
            seen[label] += 1
            multiline += continued > 0
            if len(samples) < sample_limit:
                samples.append({"unit_id": uid, "label": label, "line": line, "definition": definition, "continued_lines": continued})
        duplicate_labels.extend({"unit_id": uid, "label": label, "count": count} for label, count in seen.items() if count > 1)
    return {"total_definitions": total, "units_with_definitions": len(units_with), "duplicate_labels": duplicate_labels, "multiline_count": multiline, "label_counts": [{"label": key, "count": label_counts[key]} for key in sorted(label_counts, key=sort_key)], "samples": samples}


def _definitions(content: str) -> list[tuple[str, str, int, int]]:
    lines = content.splitlines()
    rows: list[tuple[str, str, int, int]] = []
    index = 0
    while index < len(lines):
        match = _DEF_RE.match(lines[index])
        if not match:
            index += 1
            continue
        start = index + 1
        parts = [field_value(match.group(2))]
        continued = 0
        index += 1
        while index < len(lines) and (lines[index].startswith("    ") or lines[index].startswith("\t")):
            parts.append(field_value(lines[index]))
            continued += 1
            index += 1
        rows.append((field_value(match.group(1)), " ".join(part for part in parts if part), start, continued))
    return rows
