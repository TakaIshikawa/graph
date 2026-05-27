"""Summarize Markdown task list items in units."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_TASK_RE = re.compile(r"^\s{0,3}[-+*]\s+\[(?P<mark>[ xX])]\s+")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def summarize_unit_markdown_task_lists(units: Iterable[Any], example_limit: int = 5) -> dict[str, Any]:
    total_units = units_with = total = checked = unchecked = 0
    examples: list[str] = []
    low_count: list[dict[str, int | str]] = []
    for unit in units:
        total_units += 1
        unit_checked = unit_unchecked = 0
        in_fence = False
        for line in str(get(unit, "content") or "").splitlines():
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            match = _TASK_RE.match(line)
            if not match:
                continue
            if match.group("mark").casefold() == "x":
                unit_checked += 1
            else:
                unit_unchecked += 1
        unit_total = unit_checked + unit_unchecked
        if unit_total:
            units_with += 1
            examples.append(unit_id(unit))
            low_count.append({"unit_id": unit_id(unit), "task_count": unit_total})
        total += unit_total
        checked += unit_checked
        unchecked += unit_unchecked
    examples = sorted(examples, key=sort_key)[:example_limit]
    low_count.sort(key=lambda row: (int(row["task_count"]), sort_key(row["unit_id"])))
    return {
        "total_units": total_units,
        "units_with_tasks": units_with,
        "total_task_items": total,
        "checked_task_count": checked,
        "unchecked_task_count": unchecked,
        "example_unit_ids": examples,
        "low_count_unit_samples": low_count[:example_limit],
    }
