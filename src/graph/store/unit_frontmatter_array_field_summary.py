"""Summarize list-valued fields in leading YAML frontmatter."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import yaml

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id


def summarize_unit_frontmatter_array_fields(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total_units = 0
    data: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_count": 0, "item_counts": [], "mixed_type_count": 0, "samples": []})
    for index, unit in enumerate(units):
        total_units += 1
        frontmatter = _frontmatter(str(get(unit, "content") or ""))
        for key, value in sorted(frontmatter.items(), key=lambda item: sort_key(item[0])):
            if not isinstance(value, list):
                continue
            row = data[field_value(key)]
            row["unit_count"] += 1
            row["item_counts"].append(len(value))
            if len({_kind(item) for item in value}) > 1:
                row["mixed_type_count"] += 1
            if len(row["samples"]) < limit:
                row["samples"].append({"unit_id": unit_id(unit) or str(index), "title": _title(unit)})
    rows = []
    for key in sorted(data, key=sort_key):
        counts = data[key]["item_counts"]
        rows.append(
            {
                "key": key,
                "unit_count": data[key]["unit_count"],
                "min_items": min(counts),
                "max_items": max(counts),
                "total_items": sum(counts),
                "mixed_type_count": data[key]["mixed_type_count"],
                "sample_units": data[key]["samples"],
            }
        )
    return {"total_units": total_units, "array_fields": rows}


def _frontmatter(content: str) -> dict[str, Any]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    block: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            try:
                parsed = yaml.safe_load("\n".join(block))
            except yaml.YAMLError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        block.append(line)
    return {}


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "scalar"


def _title(unit: Any) -> str:
    return field_value(get(unit, "title") or metadata(unit).get("title"))
