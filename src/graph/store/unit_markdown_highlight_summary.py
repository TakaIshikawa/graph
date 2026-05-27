"""Summarize Markdown highlight spans by source."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
_HIGHLIGHT_RE = re.compile(r"(?<!\\)==(.*?)(?<!\\)==", re.DOTALL)


def summarize_unit_markdown_highlights(units: Iterable[Any]) -> dict[str, Any]:
    total_units = 0
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_count": 0, "units_with": 0, "span_count": 0, "max": 0, "text_length": 0})
    for unit in units:
        total_units += 1
        spans = [field_value(span) for span in _HIGHLIGHT_RE.findall(_content_without_code(unit))]
        group = groups[_source(unit)]
        group["unit_count"] += 1
        group["span_count"] += len(spans)
        group["text_length"] += sum(len(span) for span in spans)
        group["max"] = max(group["max"], len(spans))
        if spans:
            group["units_with"] += 1
    rows = []
    for source, data in groups.items():
        span_count = data["span_count"]
        rows.append({
            "source": source,
            "unit_count": data["unit_count"],
            "units_with_highlights": data["units_with"],
            "highlight_span_count": span_count,
            "max_highlights_per_unit": data["max"],
            "average_highlight_text_length": round(data["text_length"] / span_count, 2) if span_count else 0,
        })
    rows.sort(key=lambda row: sort_key(row["source"]))
    return {"total_units": total_units, "sources": rows}


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
