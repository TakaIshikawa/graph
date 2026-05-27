"""Summarize tag formatting issues in unit metadata and frontmatter."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_TAG_KEYS = ("tags", "tag")
_FIELD_RE = re.compile(r"^\s*(tags?|keywords)\s*:\s*(.*?)\s*$", re.IGNORECASE)


def summarize_unit_frontmatter_tag_formats(units: Iterable[Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter({"scalar": 0, "list": 0, "empty": 0, "duplicate": 0, "whitespace": 0})
    for unit in units:
        raw = _tag_value(unit)
        tags, value_type = _tags(raw)
        normalized = [_normalize(tag) for tag in tags if _normalize(tag)]
        duplicate = len(normalized) != len(set(normalized))
        empty = raw is None or not normalized
        whitespace = any(tag != tag.strip() or re.search(r"\s{2,}", tag) for tag in tags if isinstance(tag, str))
        if value_type in ("scalar", "list"):
            totals[value_type] += 1
        if empty:
            totals["empty"] += 1
        if duplicate:
            totals["duplicate"] += 1
        if whitespace:
            totals["whitespace"] += 1
        rows.append(
            {
                "unit_id": unit_id(unit),
                "tag_value_type": value_type,
                "tag_count": len(normalized),
                "empty_tags": empty,
                "duplicate_normalized_tags": duplicate,
                "whitespace_normalization_issue": whitespace,
            }
        )
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {"total_units": len(rows), "issue_counts": dict(totals), "units": rows}


def _tag_value(unit: Any) -> Any:
    meta = metadata(unit)
    for key in _TAG_KEYS:
        value = get(unit, key)
        if value is not None:
            return value
        if key in meta:
            return meta[key]
    return _frontmatter_tags(str(get(unit, "content") or ""))


def _frontmatter_tags(content: str) -> Any:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            return None
        if match := _FIELD_RE.match(line):
            value = match.group(2).strip()
            if value.startswith("[") and value.endswith("]"):
                return [item.strip().strip("'\"") for item in value[1:-1].split(",")]
            return value
    return None


def _tags(value: Any) -> tuple[list[Any], str]:
    if value is None:
        return ([], "missing")
    if isinstance(value, list | tuple | set):
        return (list(value), "list")
    text = field_value(value)
    if "," in text:
        return ([item.strip() for item in text.split(",")], "scalar")
    return ([value], "scalar")


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", field_value(value)).strip().casefold()
