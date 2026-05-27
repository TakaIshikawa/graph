"""Summarize boolean-like metadata and frontmatter fields."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_FIELD_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$")
_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}


def summarize_unit_frontmatter_boolean_fields(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    total_units = 0
    rows: dict[str, dict[str, Any]] = defaultdict(lambda: {"true_count": 0, "false_count": 0, "string_boolean_count": 0, "invalid_boolean_like_count": 0, "examples": []})
    for index, unit in enumerate(units):
        total_units += 1
        uid = unit_id(unit) or str(index)
        for key, value in _fields(unit).items():
            state = _state(value)
            if not state:
                continue
            row = rows[key]
            if state == "true":
                row["true_count"] += 1
            elif state == "false":
                row["false_count"] += 1
            elif state == "string":
                row["string_boolean_count"] += 1
            else:
                row["invalid_boolean_like_count"] += 1
            if len(row["examples"]) < sample_limit:
                row["examples"].append({"unit_id": uid, "value": str(value)})
    return {
        "total_units": total_units,
        "boolean_fields": [{"key": key, **rows[key]} for key in sorted(rows, key=sort_key)],
    }


def _fields(unit: Any) -> dict[str, Any]:
    fields = dict(metadata(unit))
    content = str(get(unit, "content") or "")
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            match = _FIELD_RE.match(line.strip())
            if match:
                fields.setdefault(match.group(1), match.group(2).strip().strip("'\""))
    return fields


def _state(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if not isinstance(value, str):
        return ""
    text = value.strip().casefold()
    if not text:
        return ""
    if text in _TRUE | _FALSE:
        return "string"
    if text in {"truthy", "falsy", "enabled", "disabled"}:
        return "invalid"
    return ""
