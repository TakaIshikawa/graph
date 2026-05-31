"""Summarize Markdown pipe-table header names in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def summarize_unit_markdown_table_headers(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = 0
    set_counts: Counter[tuple[str, ...]] = Counter()
    name_counts: Counter[str] = Counter()
    set_units: dict[tuple[str, ...], set[str]] = {}
    name_units: dict[str, set[str]] = {}
    examples: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for unit in units:
        total += 1
        uid = unit_id(unit)
        for line_number, headers in _tables(str(get(unit, "content") or "")):
            key = tuple(headers)
            set_counts[key] += 1
            set_units.setdefault(key, set()).add(uid)
            examples.setdefault(key, [])
            if len(examples[key]) < limit:
                examples[key].append({"unit_id": uid, "line_number": line_number, "headers": list(headers)})
            for header in headers:
                name_counts[header] += 1
                name_units.setdefault(header, set()).add(uid)
    header_sets = [{"headers": list(key), "table_count": count, "unit_count": len(set_units[key]), "examples": examples[key][:limit]} for key, count in set_counts.items()]
    header_names = [{"header": key, "count": count, "unit_count": len(name_units[key])} for key, count in name_counts.items()]
    header_sets.sort(key=lambda row: (-int(row["table_count"]), sort_key("|".join(row["headers"]))))
    header_names.sort(key=lambda row: (-int(row["count"]), sort_key(row["header"])))
    return {"total_units": total, "table_count": sum(set_counts.values()), "header_sets": header_sets, "header_names": header_names}


def _tables(content: str) -> list[tuple[int, list[str]]]:
    rows = []
    lines = content.splitlines()
    in_fence = False
    active = [False] * len(lines)
    for index, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        active[index] = not in_fence
    for index in range(len(lines) - 1):
        if active[index] and active[index + 1] and "|" in lines[index] and _SEP_RE.match(lines[index + 1]):
            headers = [_normalize(cell) for cell in lines[index].strip().strip("|").split("|")]
            headers = [header for header in headers if header]
            if headers:
                rows.append((index + 1, headers))
    return rows


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", field_value(value.strip())).casefold()
