"""Summarize Markdown checklist task priority markers."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_TASK_RE = re.compile(r"^\s*[-*+]\s+\[(?P<checked>[ xX])\]\s+(?P<text>.*)$")
_PATTERNS = (
    re.compile(r"(?<!\w)#priority/(?P<value>[A-Za-z0-9_-]+)"),
    re.compile(r"\bpriority::\s*(?P<value>[A-Za-z0-9_-]+)", re.IGNORECASE),
    re.compile(r"\[priority:\s*(?P<value>[A-Za-z0-9_-]+)\]", re.IGNORECASE),
)
_BANG_RE = re.compile(r"^(?P<marker>!{1,3})(?=\s|$)")


def summarize_unit_markdown_task_priorities(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = 0
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"task_count": 0, "unit_ids": set(), "checked_count": 0, "unchecked_count": 0, "examples": []})
    for unit in units:
        total += 1
        for task in _tasks(unit):
            group = groups[task["priority"]]
            group["task_count"] += 1
            group["unit_ids"].add(task["unit_id"])
            group["checked_count" if task["checked"] else "unchecked_count"] += 1
            if len(group["examples"]) < sample_limit:
                group["examples"].append({"unit_id": task["unit_id"], "line": task["line"], "task_text": task["task_text"]})
    priorities = [
        {
            "priority": priority,
            "task_count": data["task_count"],
            "unit_count": len(data["unit_ids"]),
            "checked_count": data["checked_count"],
            "unchecked_count": data["unchecked_count"],
            "examples": data["examples"],
        }
        for priority, data in groups.items()
    ]
    priorities.sort(key=lambda row: (-row["task_count"], sort_key(row["priority"])))
    return {"total_units": total, "priorities": priorities}


def _tasks(unit: Any) -> list[dict[str, Any]]:
    rows = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _TASK_RE.match(line)
        if not match:
            continue
        marker, priority, cleaned = _priority(field_value(match.group("text")))
        if marker:
            rows.append({"unit_id": unit_id(unit), "line": line_number, "checked": match.group("checked").casefold() == "x", "priority": priority, "task_text": cleaned})
    return rows


def _priority(text: str) -> tuple[str, str, str]:
    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0), field_value(match.group("value")).casefold().replace("_", "-"), _clean(text[: match.start()] + text[match.end() :])
    match = _BANG_RE.match(text)
    if match:
        marker = match.group("marker")
        return marker, {"!": "low", "!!": "medium", "!!!": "high"}[marker], _clean(text[match.end() :])
    return "", "", text


def _clean(value: str) -> str:
    return field_value(value.strip(" -:;"))
