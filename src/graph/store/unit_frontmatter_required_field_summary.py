"""Summarize required YAML frontmatter fields on units."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, unit_id

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---", re.DOTALL)
_FIELD_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$")


def summarize_unit_frontmatter_required_fields(units: Iterable[Any], required_fields: Iterable[str], sample_limit: int = 5) -> dict[str, Any]:
    required = [field_value(field) for field in required_fields if field_value(field)]
    limit = max(0, sample_limit)
    total = 0
    rows: list[dict[str, Any]] = []
    missing_counts = {field: 0 for field in required}
    blank_counts = {field: 0 for field in required}
    present_counts = {field: 0 for field in required}

    for unit in units:
        total += 1
        fields = _frontmatter_fields(str(get(unit, "content") or ""))
        missing: list[str] = []
        blank: list[str] = []
        present: list[str] = []
        for field in required:
            if field not in fields:
                missing.append(field)
                missing_counts[field] += 1
            elif not field_value(fields[field]):
                blank.append(field)
                blank_counts[field] += 1
            else:
                present.append(field)
                present_counts[field] += 1
        if (missing or blank) and len(rows) < limit:
            rows.append({"unit_id": unit_id(unit), "missing_fields": missing, "blank_fields": blank, "present_fields": present})

    return {
        "total_units": total,
        "required_fields": required,
        "field_counts": [
            {"field": field, "missing": missing_counts[field], "blank": blank_counts[field], "present": present_counts[field]}
            for field in required
        ],
        "examples": rows,
    }


def _frontmatter_fields(content: str) -> dict[str, str]:
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        field = _FIELD_RE.match(line)
        if field:
            fields[field.group(1)] = field.group(2).strip().strip("'\"")
    return fields
