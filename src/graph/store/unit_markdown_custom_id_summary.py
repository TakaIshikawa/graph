"""Summarize Pandoc-style Markdown attribute blocks by source."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ATTR_RE = re.compile(r"\{(?P<body>[^{}\n]+)\}")


def summarize_unit_markdown_custom_ids(units: Iterable[Any]) -> dict[str, Any]:
    total_units = 0
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_count": 0, "units_with": 0, "ids": 0, "duplicates": 0, "classes": 0, "pairs": 0})
    for unit in units:
        total_units += 1
        ids, classes, pairs = _attributes(unit)
        counts = Counter(ids)
        duplicates = sum(count - 1 for count in counts.values() if count > 1)
        group = groups[_source(unit)]
        group["unit_count"] += 1
        group["ids"] += len(ids)
        group["duplicates"] += duplicates
        group["classes"] += classes
        group["pairs"] += pairs
        if ids:
            group["units_with"] += 1
    rows = [
        {
            "source": source,
            "unit_count": data["unit_count"],
            "units_with_custom_ids": data["units_with"],
            "custom_id_count": data["ids"],
            "duplicate_custom_id_count": data["duplicates"],
            "class_attribute_count": data["classes"],
            "key_value_attribute_count": data["pairs"],
        }
        for source, data in groups.items()
    ]
    rows.sort(key=lambda row: sort_key(row["source"]))
    return {"total_units": total_units, "sources": rows}


def _attributes(unit: Any) -> tuple[list[str], int, int]:
    ids: list[str] = []
    classes = 0
    pairs = 0
    in_fence = False
    for line in str(get(unit, "content") or "").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _ATTR_RE.finditer(line):
            for token in match.group("body").split():
                token = field_value(token)
                if token.startswith("#") and len(token) > 1:
                    ids.append(token[1:])
                elif token.startswith(".") and len(token) > 1:
                    classes += 1
                elif "=" in token and not token.startswith("="):
                    pairs += 1
    return ids, classes, pairs


def _source(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "source") or get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"
