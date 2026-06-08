"""CSV backlog rollup for task-like units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, inline_text, metadata, render_csv, sort_key, write_csv

_FIELDNAMES = [
    "source_project",
    "priority",
    "total_tasks",
    "open_tasks",
    "overdue_tasks",
    "due_today_tasks",
    "upcoming_tasks",
    "no_due_date_tasks",
    "blocked_tasks",
    "completed_tasks",
    "high_priority_open_tasks",
    "oldest_due_date",
]
_TASK_KEYS = {"status", "state", "priority", "due", "due_date", "deadline", "completed", "completed_at"}
_COMPLETE_STATUSES = {"complete", "completed", "done", "closed", "resolved", "cancelled", "canceled"}
_OPEN_STATUSES = {"", "false", "0", "open", "todo", "pending", "incomplete", "in progress", "blocked"}
_HIGH_PRIORITIES = {"high", "urgent", "p0", "p1", "1"}


def export_backlog_report_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    *,
    reference_date: date | str | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    today = _reference_date(reference_date)
    rows = _rows(unit_list, today)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(units: list[Mapping[str, Any] | object], today: date) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        item = _task_item(unit, today)
        if item is not None:
            groups[(item["source_project"], item["priority"])].append(item)

    rows: list[dict[str, str | int]] = []
    for (source_project, priority), tasks in sorted(groups.items(), key=lambda item: (sort_key(item[0][0]), sort_key(item[0][1]))):
        open_tasks = [task for task in tasks if not task["completed"]]
        due_dates = [task["due_date"] for task in open_tasks if task["due_date"] is not None]
        rows.append(
            {
                "source_project": source_project,
                "priority": priority,
                "total_tasks": len(tasks),
                "open_tasks": len(open_tasks),
                "overdue_tasks": sum(1 for task in open_tasks if task["due_date"] is not None and task["due_date"] < today),
                "due_today_tasks": sum(1 for task in open_tasks if task["due_date"] == today),
                "upcoming_tasks": sum(1 for task in open_tasks if task["due_date"] is not None and task["due_date"] > today),
                "no_due_date_tasks": sum(1 for task in open_tasks if task["due_date"] is None),
                "blocked_tasks": sum(1 for task in open_tasks if task["blocked"]),
                "completed_tasks": sum(1 for task in tasks if task["completed"]),
                "high_priority_open_tasks": sum(1 for task in open_tasks if task["priority"].casefold() in _HIGH_PRIORITIES),
                "oldest_due_date": min(due_dates).isoformat() if due_dates else "",
            }
        )
    return rows


def _task_item(unit: Mapping[str, Any] | object, today: date) -> dict[str, Any] | None:
    data = metadata(unit)
    due_value = _first(data, ("due_date", "due", "deadline"))
    if not _is_task_like(unit, data, due_value):
        return None
    priority = inline_text(_first(data, ("priority",))) or "unspecified"
    due_date = _parse_date(due_value)
    return {
        "source_project": field_value(get(unit, "source_project")) or "Unknown",
        "priority": priority.casefold(),
        "due_date": due_date,
        "completed": _is_completed(data, today),
        "blocked": _is_blocked(data),
    }


def _is_task_like(unit: Mapping[str, Any] | object, data: Mapping[str, Any], due_value: object) -> bool:
    if due_value not in (None, "") or any(key in data for key in _TASK_KEYS):
        return True
    source_text = " ".join([field_value(get(unit, "source_project")), field_value(get(unit, "source_entity_type"))]).casefold()
    if "task" in source_text or "todo" in source_text:
        return True
    return any(inline_text(tag).casefold() in {"task", "todo", "backlog"} for tag in _tags(unit))


def _is_completed(data: Mapping[str, Any], today: date) -> bool:
    completed = data.get("completed")
    if isinstance(completed, bool):
        return completed
    if completed is not None and inline_text(completed).casefold() not in _OPEN_STATUSES:
        return True
    status = inline_text(_first(data, ("status", "state"))).casefold()
    if status in _COMPLETE_STATUSES:
        return True
    completed_at = _parse_date(data.get("completed_at"))
    return completed_at is not None and completed_at <= today


def _is_blocked(data: Mapping[str, Any]) -> bool:
    status = inline_text(_first(data, ("status", "state"))).casefold()
    if status in {"blocked", "waiting", "on hold", "on_hold"}:
        return True
    blocked = data.get("blocked")
    if isinstance(blocked, bool):
        return blocked
    return inline_text(blocked).casefold() in {"true", "1", "yes", "y", "blocked"}


def _first(data: Mapping[str, Any], keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in data and data.get(key) not in (None, ""):
            return data.get(key)
    return None


def _parse_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = inline_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _reference_date(value: date | str | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    parsed = _parse_date(value)
    if parsed is None:
        raise ValueError("reference_date must be an ISO date")
    return parsed


def _tags(unit: Mapping[str, Any] | object) -> list[object]:
    raw = get(unit, "tags")
    if isinstance(raw, list | tuple | set):
        return list(raw)
    return []
