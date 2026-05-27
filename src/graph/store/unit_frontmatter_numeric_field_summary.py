"""Summarize numeric scalar fields in YAML frontmatter."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FIELD_RE = re.compile(r"^(\s*)([A-Za-z0-9_-]+)\s*:\s*(.*)$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+)$")


def summarize_unit_frontmatter_numeric_fields(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = 0
    fields: Counter[str] = Counter()
    types: Counter[str] = Counter()
    negatives: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for unit in units:
        total += 1
        numeric = _numeric_fields(str(get(unit, "content") or ""))
        if numeric:
            units_with += 1
        for key, value, kind in numeric:
            fields[key] += 1
            types[kind] += 1
            if float(value) < 0:
                negatives[key] += 1
            if len(samples) < limit:
                samples.append({"unit_id": unit_id(unit), "field": key, "value": value, "type": kind})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["field"])))
    return {
        "total_units": total,
        "units_with_numeric_frontmatter": units_with,
        "field_counts": {key: fields[key] for key in sorted(fields, key=sort_key)},
        "type_counts": {key: types[key] for key in sorted(types, key=sort_key)},
        "negative_value_counts": {key: negatives[key] for key in sorted(negatives, key=sort_key)},
        "samples": samples[:limit],
    }


def _numeric_fields(content: str) -> list[tuple[str, str, str]]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    block: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            return _parse(block)
        block.append(line)
    return []


def _parse(lines: list[str]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    stack: list[tuple[int, str]] = []
    for line in lines:
        match = _FIELD_RE.match(line)
        if not match or line.lstrip().startswith("-"):
            continue
        indent_text, key, raw = match.groups()
        indent = len(indent_text.replace("\t", "    "))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        value = _strip_comment(raw).strip().strip("\"'")
        if value == "" or value.startswith(("[", "{")):
            stack.append((indent, key))
            continue
        kind = "integer" if _INT_RE.match(value) else "float" if _FLOAT_RE.match(value) else ""
        if kind:
            rows.append((".".join([*(path for _, path in stack), key]), value, kind))
    return rows


def _strip_comment(value: str) -> str:
    quote = ""
    for index, char in enumerate(value):
        if char in {"'", '"'} and (index == 0 or value[index - 1] != "\\"):
            quote = "" if quote == char else char if not quote else quote
        if char == "#" and not quote and (index == 0 or value[index - 1].isspace()):
            return value[:index]
    return value
