"""Summarize Markdown abbreviation definitions."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_ABBR_RE = re.compile(r"^\s*\*\[(?P<abbr>[^\]]+)\]:\s*(?P<expansion>.+?)\s*$")


def summarize_unit_markdown_abbreviations(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = 0
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"definition_count": 0, "unit_ids": set(), "expansions": set(), "examples": []})
    for unit in units:
        total += 1
        for line, abbr, expansion in _definitions(unit):
            group = groups[abbr]
            group["definition_count"] += 1
            group["unit_ids"].add(unit_id(unit))
            group["expansions"].add(expansion)
            if len(group["examples"]) < sample_limit:
                group["examples"].append({"unit_id": unit_id(unit), "line": line, "expansion": expansion})
    rows = [
        {
            "abbreviation": abbr,
            "definition_count": data["definition_count"],
            "unit_count": len(data["unit_ids"]),
            "distinct_expansions": sorted(data["expansions"], key=sort_key),
            "examples": data["examples"],
        }
        for abbr, data in groups.items()
    ]
    rows.sort(key=lambda row: (-row["definition_count"], sort_key(row["abbreviation"])))
    return {"total_units": total, "abbreviations": rows}


def _definitions(unit: Any) -> list[tuple[int, str, str]]:
    rows = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _ABBR_RE.match(line)
        if match:
            rows.append((line_number, field_value(match.group("abbr")), field_value(match.group("expansion"))))
    return rows
