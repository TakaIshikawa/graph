"""Summarize empty array assignments in YAML frontmatter."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get

_FIELD_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.*?)\s*(?:#.*)?$")


def summarize_unit_frontmatter_empty_arrays(units: Iterable[Any]) -> dict[str, Any]:
    total = units_with = 0
    key_counts: Counter[str] = Counter()
    syntax_counts: Counter[str] = Counter()
    for unit in units:
        found = False
        for key, syntax in _empty_arrays(str(get(unit, "content") or "")):
            total += 1; found = True; key_counts[key] += 1; syntax_counts[syntax] += 1
        if found:
            units_with += 1
    return {"total_empty_arrays": total, "units_with_empty_arrays": units_with, "key_counts": dict(sorted(key_counts.items())), "syntax_counts": dict(sorted(syntax_counts.items()))}


def _empty_arrays(content: str) -> list[tuple[str, str]]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    frontmatter: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            return _scan(frontmatter)
        frontmatter.append(line)
    return []


def _scan(lines: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        if not (match := _FIELD_RE.match(line)):
            continue
        key, value = match.groups()
        if value.strip() == "[]":
            rows.append((key, "inline"))
        elif value.strip() == "" and _is_empty_block(lines, index):
            rows.append((key, "block"))
    return rows


def _is_empty_block(lines: list[str], index: int) -> bool:
    current_indent = len(lines[index]) - len(lines[index].lstrip())
    next_index = index + 1
    while next_index < len(lines) and not lines[next_index].strip():
        next_index += 1
    if next_index >= len(lines):
        return True
    next_line = lines[next_index]
    next_indent = len(next_line) - len(next_line.lstrip())
    return next_indent <= current_indent and not next_line.lstrip().startswith("-")
