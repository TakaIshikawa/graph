"""Summarize Markdown task inventory across units."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TASK_RE = re.compile(r"^\s{0,3}[-+*]\s+\[(?P<mark>.)]\s+(?P<body>.*)$")
_TAG_RE = re.compile(r"(?<!\w)#([A-Za-z][\w/-]*)")
_DATE_RE = re.compile(r"\b(?:due[:=]\s*)?(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)


def summarize_unit_task_inventory(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = units_with = total = completed = open_tasks = unknown = 0
    tags: Counter[str] = Counter()
    overdue: list[dict[str, Any]] = []
    today = date.today()
    for unit in units:
        total_units += 1
        unit_found = False
        in_fence = False
        for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            match = _TASK_RE.match(line)
            if not match:
                continue
            unit_found = True
            total += 1
            mark = match.group("mark")
            body = match.group("body")
            if mark.casefold() == "x":
                completed += 1
            elif mark == " ":
                open_tasks += 1
            else:
                unknown += 1
            tags.update(tag.casefold() for tag in _TAG_RE.findall(body))
            due = _due_date(body)
            if due and due < today and len(overdue) < sample_limit:
                overdue.append({"unit_id": unit_id(unit), "line": line_number, "due_date": due.isoformat(), "task": body})
        if unit_found:
            units_with += 1
    top_tags = [{"tag": tag, "count": tags[tag]} for tag in sorted(tags, key=lambda item: (-tags[item], sort_key(item)))[:sample_limit]]
    return {"total_units": total_units, "total_tasks": total, "completed_count": completed, "open_count": open_tasks, "unknown_count": unknown, "units_with_tasks": units_with, "top_tags": top_tags, "overdue_due_date_samples": overdue}


def _due_date(text: str) -> date | None:
    match = _DATE_RE.search(text)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None
