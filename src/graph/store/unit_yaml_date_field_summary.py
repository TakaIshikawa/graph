"""Summarize date-like fields declared in markdown frontmatter."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_DEFAULT_FIELDS = ("created", "updated", "published", "date")
_KEY_VALUE_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*:\s*(.*)$")


def summarize_unit_yaml_date_fields(units: Iterable[Any], field_names: Iterable[str] | None = None) -> dict[str, Any]:
    fields = {field.casefold() for field in (field_names or _DEFAULT_FIELDS)}
    counts = {field: {"valid": 0, "missing": 0, "invalid": 0} for field in sorted(fields, key=sort_key)}
    invalid_examples = []
    total_units = 0
    for index, unit in enumerate(units):
        total_units += 1
        data = _frontmatter(_content(unit))
        normalized = {str(key).casefold(): value for key, value in data.items()}
        uid = unit_id(unit) or str(index)
        for field in counts:
            if field not in normalized or normalized[field] in (None, ""):
                counts[field]["missing"] += 1
            elif _valid_iso_date(normalized[field]):
                counts[field]["valid"] += 1
            else:
                counts[field]["invalid"] += 1
                invalid_examples.append({"unit_id": uid, "field": field, "value": str(normalized[field])})
    invalid_examples.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["field"])))
    return {"total_units": total_units, "field_counts": counts, "invalid_examples": invalid_examples}


def _content(unit: Any) -> str:
    value = get(unit, "content")
    if value in (None, ""):
        value = metadata(unit).get("content")
    return "" if value is None else str(value)


def _frontmatter(content: str) -> dict[str, Any]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            data = {}
            for line in lines[1:index]:
                match = _KEY_VALUE_RE.match(line)
                if match:
                    data[match.group(1)] = match.group(2).strip().strip("\"'")
            return data
    return {}


def _valid_iso_date(value: Any) -> bool:
    if isinstance(value, date):
        return True
    text = str(value)
    try:
        return date.fromisoformat(text).isoformat() == text
    except ValueError:
        return False
