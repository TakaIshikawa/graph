"""CSV export for unit reading progress metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.export._report_csv import (
    field_value,
    get,
    metadata,
    render_csv,
    sort_key,
    unit_id,
    write_csv,
)

_FIELDNAMES = [
    "unit_id",
    "title",
    "status",
    "pages_read",
    "total_pages",
    "progress_percent",
    "started_at",
    "completed_at",
]


def export_unit_reading_progress_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-unit reading progress from unit metadata."""
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {
        "path": output_path,
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": bytes_written,
    }


def _row(unit: Mapping[str, Any] | object) -> dict[str, str]:
    data = metadata(unit)
    pages_read = _number(
        data.get("pages_read") or data.get("page_read") or data.get("current_page")
    )
    total_pages = _number(data.get("total_pages") or data.get("pages") or data.get("page_count"))
    status = field_value(data.get("status") or data.get("reading_status"))
    progress = _progress_percent(
        data.get("progress") or data.get("progress_percent"), pages_read, total_pages, status
    )
    return {
        "unit_id": unit_id(unit),
        "title": field_value(get(unit, "title")),
        "status": status,
        "pages_read": _format_number(pages_read),
        "total_pages": _format_number(total_pages),
        "progress_percent": _format_percent(progress),
        "started_at": _date_text(data.get("started_at") or data.get("start_date")),
        "completed_at": _date_text(
            data.get("completed_at") or data.get("completion_date") or data.get("finished_at")
        ),
    }


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = field_value(value).removesuffix("%").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _progress_percent(
    value: object, pages_read: float | None, total_pages: float | None, status: str
) -> float | None:
    explicit = _number(value)
    if explicit is not None:
        return _clamp(explicit * 100 if 0 <= explicit <= 1 else explicit)
    if pages_read is not None and total_pages and total_pages > 0:
        return _clamp((pages_read / total_pages) * 100)
    if status.casefold() in {"completed", "complete", "done", "finished", "read"}:
        return 100.0
    return None


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def _format_percent(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


def _date_text(value: object) -> str:
    text = field_value(value)
    if not text:
        return ""
    candidate = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(candidate).date().isoformat()
    except ValueError:
        return text
