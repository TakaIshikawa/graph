"""Summarize Markdown strikethrough spans by source."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id, field_value

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
_STRIKE_RE = re.compile(r"(?<!\\)~~(.*?)(?<!\\)~~", re.DOTALL)


def summarize_unit_markdown_strikethrough(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = 0
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_count": 0, "units": set(), "span_count": 0, "max": 0})
    for unit in units:
        total_units += 1
        source = _source(unit)
        spans = _spans(unit)
        group = groups[source]
        group["unit_count"] += 1
        group["span_count"] += len(spans)
        group["max"] = max(group["max"], len(spans))
        if spans:
            group["units"].add(unit_id(unit))
    rows = [
        {
            "source": source,
            "unit_count": data["unit_count"],
            "units_with_strikethrough": len(data["units"]),
            "strikethrough_span_count": data["span_count"],
            "max_spans_per_unit": data["max"],
            "sample_units": sorted(data["units"], key=sort_key)[:sample_limit],
        }
        for source, data in groups.items()
    ]
    rows.sort(key=lambda row: sort_key(row["source"]))
    return {"total_units": total_units, "sources": rows}


def _spans(unit: Any) -> list[str]:
    return [field_value(match) for match in _STRIKE_RE.findall(_content_without_code(unit))]


def _content_without_code(unit: Any) -> str:
    lines: list[str] = []
    in_fence = False
    for line in str(get(unit, "content") or "").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(_INLINE_CODE_RE.sub("", line))
    return "\n".join(lines)


def _source(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "source") or get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"
