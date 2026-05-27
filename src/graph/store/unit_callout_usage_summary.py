"""Summarize Obsidian-style callout usage by source."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_CALLOUT_RE = re.compile(r"^\s*>\s*\[!([A-Za-z][\w-]*)\]([+-])?", re.IGNORECASE)


def summarize_unit_callout_usage(units: Iterable[Any]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    total_units = 0
    for unit in units:
        total_units += 1
        grouped[_source(unit)].append(unit)

    rows = [_row(source, grouped[source]) for source in sorted(grouped, key=sort_key)]
    return {"total_units": total_units, "rows": rows, "source_summaries": rows}


def _row(source: str, units: list[Any]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    units_with_callouts = callout_count = folded_count = max_per_unit = 0
    for unit in units:
        callouts = _callouts(_content(unit))
        if callouts:
            units_with_callouts += 1
        callout_count += len(callouts)
        folded_count += sum(1 for _kind, fold in callouts if fold)
        max_per_unit = max(max_per_unit, len(callouts))
        type_counts.update(kind for kind, _fold in callouts)

    most_common = ""
    if type_counts:
        most_common = sorted(type_counts.items(), key=lambda item: (-item[1], sort_key(item[0])))[0][0]
    return {
        "source": source,
        "unit_count": len(units),
        "units_with_callouts": units_with_callouts,
        "callout_count": callout_count,
        "most_common_callout_type": most_common,
        "folded_callout_count": folded_count,
        "max_callouts_per_unit": max_per_unit,
    }


def _callouts(content: str) -> list[tuple[str, bool]]:
    rows: list[tuple[str, bool]] = []
    for line in content.splitlines():
        match = _CALLOUT_RE.match(line)
        if match:
            rows.append((match.group(1).casefold(), bool(match.group(2))))
    return rows


def _source(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"


def _content(unit: Any) -> str:
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)
