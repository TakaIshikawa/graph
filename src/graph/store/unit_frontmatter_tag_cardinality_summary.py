"""Summarize YAML frontmatter tag cardinality across units."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_TAG_FIELD_RE = re.compile(r"^\s*tags\s*:\s*(.*?)\s*$", re.IGNORECASE)


def summarize_unit_frontmatter_tag_cardinality(units: Iterable[Any], high_cardinality_threshold: int = 5, top_limit: int = 10) -> dict[str, Any]:
    unit_list = list(units)
    tag_counts: Counter[str] = Counter()
    no_tags: list[str] = []
    high: list[dict[str, Any]] = []
    duplicate: list[dict[str, Any]] = []
    for unit in unit_list:
        tags = _frontmatter_tags(str(get(unit, "content") or ""))
        normalized = [_normalize(tag) for tag in tags if _normalize(tag)]
        tag_counts.update(normalized)
        uid = unit_id(unit)
        if not normalized:
            no_tags.append(uid)
        if len(set(normalized)) >= high_cardinality_threshold and normalized:
            high.append({"unit_id": uid, "tag_count": len(set(normalized))})
        duplicates = sorted({tag for tag in normalized if normalized.count(tag) > 1}, key=sort_key)
        if duplicates:
            duplicate.append({"unit_id": uid, "duplicate_tags": duplicates})
    top_tags = [{"tag": tag, "unit_count": count} for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], sort_key(item[0])))[: max(0, top_limit)]]
    return {
        "total_units": len(unit_list),
        "distinct_normalized_tag_count": len(tag_counts),
        "units_with_no_tags": sorted(no_tags, key=sort_key),
        "high_cardinality_units": sorted(high, key=lambda row: (-row["tag_count"], sort_key(row["unit_id"]))),
        "duplicate_tag_units": sorted(duplicate, key=lambda row: sort_key(row["unit_id"])),
        "top_tags": top_tags,
    }


def _frontmatter_tags(content: str) -> list[str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    tags: list[str] = []
    in_tags = False
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if in_tags:
            if match := re.match(r"^\s*-\s*(.+?)\s*$", line):
                tags.append(match.group(1).strip("'\""))
                continue
            if line.strip() and not line.startswith((" ", "\t")):
                in_tags = False
        if match := _TAG_FIELD_RE.match(line):
            value = match.group(1).strip()
            if not value:
                in_tags = True
            elif value.startswith("[") and value.endswith("]"):
                tags.extend(item.strip().strip("'\"") for item in value[1:-1].split(","))
            else:
                tags.extend(item.strip().strip("'\"") for item in value.split(","))
    return tags


def _normalize(value: str) -> str:
    return field_value(value).casefold()
