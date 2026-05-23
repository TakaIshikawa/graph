"""CSV export for stored source robots/crawl policy metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, normalized_key, parse_datetime, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["source_id", "robots_allowed", "crawl_delay", "robots_source", "policy_checked_at"]
_ALIASES = {
    "robots_allowed": ("robots_allowed", "crawl_allowed", "allowed_by_robots", "robots_allow"),
    "crawl_delay": ("crawl_delay", "robots_crawl_delay", "delay"),
    "robots_source": ("robots_source", "robots_url", "policy_source"),
    "policy_checked_at": ("policy_checked_at", "robots_checked_at", "checked_at"),
}
_UNKNOWN = "unknown"


def export_source_robots_policy_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per source with stored robots policy values."""
    source_list = list(sources)
    rows = [_row(source) for source in source_list]
    rows.sort(key=lambda row: sort_key(row["source_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(source: Mapping[str, Any] | object) -> dict[str, str]:
    checked_at = _value(source, "policy_checked_at")
    parsed = parse_datetime(checked_at)
    return {
        "source_id": source_id(source),
        "robots_allowed": _bool_or_unknown(_value(source, "robots_allowed")),
        "crawl_delay": _value(source, "crawl_delay") or _UNKNOWN,
        "robots_source": _value(source, "robots_source") or _UNKNOWN,
        "policy_checked_at": parsed.isoformat() if parsed else checked_at or _UNKNOWN,
    }


def _value(source: Mapping[str, Any] | object, field: str) -> str:
    for alias in _ALIASES[field]:
        text = field_value(get(source, alias))
        if text:
            return text
    alias_keys = {normalized_key(alias) for alias in _ALIASES[field]}
    for key, value in metadata(source).items():
        if normalized_key(key) in alias_keys and field_value(value):
            return field_value(value)
    return ""


def _bool_or_unknown(value: object) -> str:
    text = field_value(value).casefold()
    if text in {"true", "1", "yes", "allowed", "allow"}:
        return "true"
    if text in {"false", "0", "no", "blocked", "disallowed", "deny"}:
        return "false"
    return _UNKNOWN
