"""CSV export for source robots policy checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["source_id", "source_name", "robots_allowed", "crawl_delay", "user_agent", "policy_url", "checked_at"]
_ALIASES = {
    "source_name": ("name", "title", "source_name"),
    "robots_allowed": ("robots_allowed", "allowed_by_robots", "crawl_allowed"),
    "crawl_delay": ("crawl_delay", "robots_crawl_delay"),
    "user_agent": ("user_agent", "robots_user_agent"),
    "policy_url": ("policy_url", "robots_url", "robots_source"),
    "checked_at": ("checked_at", "robots_checked_at", "policy_checked_at"),
}


def export_source_robot_policy_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source) for source in source_list]
    rows.sort(key=lambda row: sort_key(row["source_id"] or row["source_name"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(source: Mapping[str, Any] | object) -> dict[str, str]:
    checked = _value(source, "checked_at")
    parsed = parse_datetime(checked)
    return {
        "source_id": source_id(source),
        "source_name": _value(source, "source_name"),
        "robots_allowed": _bool_value(_value(source, "robots_allowed")),
        "crawl_delay": _value(source, "crawl_delay"),
        "user_agent": _value(source, "user_agent"),
        "policy_url": _value(source, "policy_url"),
        "checked_at": parsed.isoformat() if parsed else checked,
    }


def _value(source: Mapping[str, Any] | object, field: str) -> str:
    data = metadata(source)
    for alias in _ALIASES[field]:
        text = field_value(get(source, alias))
        if text:
            return text
    for alias in _ALIASES[field]:
        text = field_value(data.get(alias))
        if text:
            return text
    return ""


def _bool_value(value: str) -> str:
    text = value.casefold()
    if text in {"1", "true", "yes", "allowed", "allow"}:
        return "true"
    if text in {"0", "false", "no", "blocked", "disallowed", "deny"}:
        return "false"
    return value
