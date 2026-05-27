"""Summarize Markdown horizontal rules by source."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_RULE_RE = re.compile(r"^\s{0,3}(?P<marker>(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$")
_MARKER_ORDER = "-*_"


def summarize_unit_markdown_horizontal_rules(units: Iterable[Any]) -> dict[str, Any]:
    total_units = 0
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_count": 0, "units_with": 0, "rules": 0, "max": 0, "markers": Counter()})
    for unit in units:
        total_units += 1
        markers = _markers(unit)
        group = groups[_source(unit)]
        group["unit_count"] += 1
        group["rules"] += len(markers)
        group["max"] = max(group["max"], len(markers))
        group["markers"].update(markers)
        if markers:
            group["units_with"] += 1
    rows = []
    for source, data in groups.items():
        markers: Counter[str] = data["markers"]
        marker = max(_MARKER_ORDER, key=lambda value: (markers[value], -_MARKER_ORDER.index(value))) if markers else ""
        rows.append({
            "source": source,
            "unit_count": data["unit_count"],
            "units_with_horizontal_rules": data["units_with"],
            "horizontal_rule_count": data["rules"],
            "most_common_rule_marker": marker,
            "max_rules_per_unit": data["max"],
        })
    rows.sort(key=lambda row: sort_key(row["source"]))
    return {"total_units": total_units, "sources": rows}


def _markers(unit: Any) -> list[str]:
    markers: list[str] = []
    in_fence = False
    in_frontmatter = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line_number == 1 and line.strip() == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if line.strip() == "---":
                in_frontmatter = False
            continue
        match = _RULE_RE.match(line)
        if match:
            markers.append(match.group("marker").strip()[0])
    return markers


def _source(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "source") or get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"
