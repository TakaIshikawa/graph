"""Summarize Obsidian callout fold states in Markdown content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_CALLOUT_RE = re.compile(r"^\s*>\s*\[!([A-Za-z0-9_-]+)]([+-]?)")


def summarize_unit_markdown_callout_fold_states(units: Iterable[Any]) -> dict[str, Any]:
    total = units_with = callout_count = 0
    states: Counter[str] = Counter()
    types: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        unit_count = 0
        unit_states: Counter[str] = Counter()
        unit_types: Counter[str] = Counter()
        for line in str(get(unit, "content") or "").splitlines():
            match = _CALLOUT_RE.match(line)
            if not match:
                continue
            state = "open" if match.group(2) == "+" else "closed" if match.group(2) == "-" else "none"
            callout_type = match.group(1).casefold()
            unit_count += 1
            unit_states[state] += 1
            unit_types[callout_type] += 1
        if unit_count:
            units_with += 1
            callout_count += unit_count
            states.update(unit_states)
            types.update(unit_types)
            rows.append({"unit_id": uid, "callout_count": unit_count, "open_count": unit_states["open"], "closed_count": unit_states["closed"], "none_count": unit_states["none"]})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {
        "total_units": total,
        "units_with_callouts": units_with,
        "callout_count": callout_count,
        "fold_states": {"open": states["open"], "closed": states["closed"], "none": states["none"]},
        "types": {key: types[key] for key in sorted(types, key=sort_key)},
        "units": rows,
    }
