"""Markdown task board export helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")
_COMPLETED_STATUSES = {
    "complete",
    "completed",
    "done",
    "closed",
    "resolved",
    "cancelled",
    "canceled",
}
_TASK_SOURCE_MARKERS = ("task", "todo", "reminder")
_SECTIONS = ("Overdue", "Due Today", "Upcoming", "No Due Date", "Completed")


def export_task_board_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    status_keys: Sequence[str] = ("status", "completed"),
    due_date_keys: Sequence[str] = ("due", "due_date", "deadline"),
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown task board."""
    today = datetime.now(timezone.utc).date()
    tasks = [
        _task_entry(unit, today, status_keys=status_keys, due_date_keys=due_date_keys)
        for unit in units
    ]
    tasks = [task for task in tasks if task is not None]
    text = _render_board(tasks, today)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "tasks_exported": len(tasks),
        "bytes_written": output_path.stat().st_size,
    }


def _task_entry(
    unit: KnowledgeUnit,
    today: date,
    *,
    status_keys: Sequence[str],
    due_date_keys: Sequence[str],
) -> dict[str, Any] | None:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    due_value = _first_metadata(metadata, due_date_keys)
    completed = _is_completed(metadata, status_keys)

    if not _is_task_like(unit, metadata, due_value, status_keys):
        return None

    due_date = _parse_date(due_value)
    if completed:
        section = "Completed"
    elif due_date is None:
        section = "No Due Date"
    elif due_date < today:
        section = "Overdue"
    elif due_date == today:
        section = "Due Today"
    else:
        section = "Upcoming"

    return {
        "section": section,
        "due_date": due_date,
        "due_text": _date_text(due_value, due_date),
        "title": _inline_text(unit.title) or _inline_text(unit.id) or "Untitled",
        "source_project": _field_value(unit.source_project) or "_None_",
        "tags": _unit_tags(unit),
        "snippet": _snippet(unit.content),
        "id": _inline_text(unit.id),
    }


def _render_board(tasks: list[dict[str, Any]], today: date) -> str:
    grouped = {section: [] for section in _SECTIONS}
    for task in tasks:
        grouped[task["section"]].append(task)

    lines = [
        "# Task Board",
        "",
        f"_As of {today.isoformat()}_",
        "",
    ]
    for section in _SECTIONS:
        section_tasks = sorted(grouped[section], key=_task_sort_key)
        lines.extend([f"## {section}", ""])
        if section_tasks:
            for task in section_tasks:
                lines.append(_task_line(task))
        else:
            lines.append("_No tasks._")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _task_line(task: Mapping[str, Any]) -> str:
    tags = ", ".join(task["tags"]) if task["tags"] else "_None_"
    due = task["due_text"] or "_None_"
    snippet = task["snippet"] or "_No content._"
    return (
        f"- **{_inline_markdown(task['title'])}**"
        f" - source: {_inline_markdown(task['source_project'])}"
        f"; due: {_inline_markdown(due)}"
        f"; tags: {_inline_markdown(tags)}"
        f"; {_inline_markdown(snippet)}"
    )


def _is_task_like(
    unit: KnowledgeUnit,
    metadata: Mapping[str, Any],
    due_value: Any,
    status_keys: Sequence[str],
) -> bool:
    if due_value not in (None, ""):
        return True
    if any(key in metadata for key in status_keys):
        return True
    source_text = " ".join(
        [_field_value(unit.source_project), _inline_text(unit.source_entity_type)]
    ).casefold()
    if any(marker in source_text for marker in _TASK_SOURCE_MARKERS):
        return True
    return any(_inline_text(tag).casefold() in {"task", "todo"} for tag in unit.tags)


def _is_completed(metadata: Mapping[str, Any], status_keys: Sequence[str]) -> bool:
    for key in status_keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        text = _inline_text(value).casefold()
        if text in _COMPLETED_STATUSES:
            return True
        if text in {"false", "0", "open", "todo", "pending", "incomplete"}:
            return False
    return False


def _first_metadata(metadata: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in metadata and metadata.get(key) not in (None, ""):
            return metadata.get(key)
    return None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _date_text(value: Any, parsed: date | None) -> str:
    if parsed is not None:
        return parsed.isoformat()
    return _inline_text(value)


def _task_sort_key(task: Mapping[str, Any]) -> tuple[str, str, str]:
    due = task.get("due_date")
    due_key = due.isoformat() if isinstance(due, date) else "9999-12-31"
    return (due_key, _inline_text(task.get("title")).casefold(), _inline_text(task.get("id")))


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted(
        {_inline_text(tag) for tag in unit.tags if _inline_text(tag)},
        key=lambda tag: (tag.casefold(), tag),
    )


def _snippet(value: object, *, length: int = 120) -> str:
    text = _inline_text(value)
    if len(text) <= length:
        return text
    return text[: length - 3].rstrip() + "..."


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _inline_markdown(value: object) -> str:
    return (
        _inline_text(value)
        .replace("\\", r"\\")
        .replace("*", r"\*")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )
