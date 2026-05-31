"""Summarize date ranges declared in unit frontmatter or metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_PAIRS = (("start", "end"), ("start_date", "end_date"), ("from", "to"), ("valid_from", "valid_to"))
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---", re.DOTALL)
_FIELD_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$")


def summarize_unit_frontmatter_date_ranges(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = 0
    grouped = {f"{left}/{right}": {"field_pair": f"{left}/{right}", "complete": 0, "missing_start": 0, "missing_end": 0, "inverted": 0, "examples": []} for left, right in _PAIRS}
    for unit in units:
        total += 1
        fields = _fields(unit)
        for left, right in _PAIRS:
            if left not in fields and right not in fields:
                continue
            row = grouped[f"{left}/{right}"]
            start_value = field_value(fields.get(left, ""))
            end_value = field_value(fields.get(right, ""))
            start_date = _parse_date(start_value)
            end_date = _parse_date(end_value)
            if not start_value:
                status = "missing_start"
            elif not end_value:
                status = "missing_end"
            else:
                status = "complete"
            row[status] += 1
            if start_date and end_date and start_date > end_date:
                row["inverted"] += 1
            if len(row["examples"]) < limit:
                row["examples"].append({"unit_id": unit_id(unit), "field_pair": row["field_pair"], "start_value": start_value, "end_value": end_value})
    rows = [row for row in grouped.values() if row["complete"] or row["missing_start"] or row["missing_end"] or row["inverted"]]
    rows.sort(key=lambda row: sort_key(row["field_pair"]))
    return {"total_units": total, "date_ranges": rows}


def _fields(unit: Any) -> dict[str, Any]:
    fields = {str(key).casefold().replace("-", "_"): value for key, value in metadata(unit).items()}
    match = _FRONTMATTER_RE.match(str(get(unit, "content") or ""))
    if match:
        for line in match.group(1).splitlines():
            field = _FIELD_RE.match(line)
            if field:
                fields[field.group(1).casefold().replace("-", "_")] = field.group(2).strip().strip("'\"")
    return fields


def _parse_date(value: Any) -> date | None:
    text = field_value(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None
