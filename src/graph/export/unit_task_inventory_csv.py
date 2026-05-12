"""CSV export for task-like unit inventory rows."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "source_entity_type",
    "status",
    "priority",
    "due_date",
    "completed",
    "completed_at",
    "assignee_count",
    "assignees",
    "checklist_item_count",
    "checklist_completed_count",
]
_TASK_KEYS = {
    "status",
    "state",
    "priority",
    "due",
    "due_date",
    "completed",
    "completed_at",
    "assignee",
    "assignees",
    "checklist_items",
    "checklist_completed_count",
}
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_task_inventory_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write normalized task inventory rows for units with task metadata."""
    unit_list = list(units)
    rows = _task_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _task_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
        if not _has_task_metadata(metadata):
            continue

        checklist_items = _list_value(metadata.get("checklist_items"))
        checklist_completed_count = _int_value(metadata.get("checklist_completed_count"))
        if checklist_completed_count is None:
            checklist_completed_count = _derived_completed_count(checklist_items)

        assignees = _assignees(metadata)
        completed_at = _date_value(metadata.get("completed_at"))
        rows.append(
            {
                "unit_id": _field_value(unit.id),
                "title": _field_value(unit.title),
                "source_project": _field_value(unit.source_project) or "Unknown",
                "source_entity_type": _field_value(unit.source_entity_type) or "Unknown",
                "status": _normalized_text(_first_present(metadata, ("status", "state"))),
                "priority": _normalized_text(metadata.get("priority")),
                "due_date": _date_text(_first_present(metadata, ("due_date", "due"))),
                "completed": _completed_text(metadata.get("completed"), completed_at),
                "completed_at": completed_at.isoformat() if completed_at else "",
                "assignee_count": len(assignees),
                "assignees": "; ".join(assignees),
                "checklist_item_count": len(checklist_items),
                "checklist_completed_count": checklist_completed_count,
            }
        )

    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["title"])))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _has_task_metadata(metadata: dict) -> bool:
    return any(_inline_text(key) in _TASK_KEYS for key in metadata)


def _first_present(metadata: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in metadata:
            return metadata.get(key)
    return None


def _assignees(metadata: dict) -> list[str]:
    values: list[object] = []
    if "assignees" in metadata:
        values.extend(_list_value(metadata.get("assignees")))
    elif "assignee" in metadata:
        values.extend(_list_value(metadata.get("assignee")))
    return sorted({_inline_text(value) for value in values if _inline_text(value)}, key=_sort_key)


def _list_value(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    return [value]


def _derived_completed_count(items: list[object]) -> int:
    return sum(1 for item in items if _item_completed(item))


def _item_completed(item: object) -> bool:
    if isinstance(item, dict):
        for key in ("completed", "done", "checked"):
            if key in item:
                return _truthy(item.get(key))
        status = _inline_text(item.get("status")).casefold()
        return status in {"complete", "completed", "done", "closed"}
    return False


def _completed_text(value: object, completed_at: date | None) -> str:
    if value is None:
        return "true" if completed_at else "false"
    return "true" if _truthy(value) else "false"


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0
    text = _inline_text(value).casefold()
    return text in {"1", "true", "yes", "y", "done", "complete", "completed", "closed"}


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = _inline_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _date_text(value: object) -> str:
    parsed = _date_value(value)
    return parsed.isoformat() if parsed else _normalized_text(value)


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _inline_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _normalized_text(value: object) -> str:
    return _inline_text(getattr(value, "value", value)).casefold()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
