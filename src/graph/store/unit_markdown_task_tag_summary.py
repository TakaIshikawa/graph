"""Summarize tags attached to Markdown task list items."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TASK_RE = re.compile(r"^\s{0,3}[-*+]\s+\[([ xX])]\s+(.*)$")
_TAG_RE = re.compile(r"(?<!\w)([#@+][A-Za-z0-9][A-Za-z0-9_-]*)")


def summarize_unit_markdown_task_tags(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = task_count = 0
    grouped: dict[str, dict[str, Any]] = {}
    for unit in units:
        total += 1
        uid = unit_id(unit)
        seen: set[str] = set()
        for line_number, checked, text, tags in _tasks(str(get(unit, "content") or "")):
            task_count += 1
            for tag in tags:
                row = grouped.setdefault(tag, {"marker": tag, "task_count": 0, "unit_ids": set(), "checked_count": 0, "unchecked_count": 0, "examples": []})
                row["task_count"] += 1
                row["checked_count" if checked else "unchecked_count"] += 1
                seen.add(tag)
                if len(row["examples"]) < limit:
                    row["examples"].append({"unit_id": uid, "line_number": line_number, "task_text": field_value(text)})
        for tag in seen:
            grouped[tag]["unit_ids"].add(uid)
    rows = [
        {"marker": row["marker"], "task_count": row["task_count"], "unit_count": len(row["unit_ids"]), "checked_count": row["checked_count"], "unchecked_count": row["unchecked_count"], "examples": row["examples"][:limit]}
        for row in grouped.values()
    ]
    rows.sort(key=lambda row: (-int(row["task_count"]), sort_key(row["marker"])))
    return {"total_units": total, "task_count": task_count, "tags": rows}


def _tasks(content: str) -> list[tuple[int, bool, str, list[str]]]:
    rows = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _TASK_RE.match(line)
        if match:
            text = match.group(2)
            rows.append((line_number, match.group(1).casefold() == "x", text, _TAG_RE.findall(text)))
    return rows
