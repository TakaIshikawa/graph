"""Summarize source Last-Modified header coverage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, sort_key, source_id


def summarize_source_last_modified(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["last_modified"]]
    parseable = [row for row in present if row["parsed"]]
    invalid = [row for row in present if not row["parsed"]]
    parsed_values = sorted(row["parsed"] for row in parseable if row["parsed"])
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "present_count": len(present),
        "missing_count": len(source_list) - len(present),
        "parseable_count": len(parseable),
        "invalid_count": len(invalid),
        "oldest": parsed_values[0].isoformat() if parsed_values else "",
        "newest": parsed_values[-1].isoformat() if parsed_values else "",
        "samples": sorted((row for row in rows if not row["last_modified"] or not row["parsed"]), key=lambda row: sort_key(row["source_id"]))[:limit],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _last_modified(source)
    return {"source_id": source_id(source) or str(index), "last_modified": value, "parsed": _parse(value)}


def _last_modified(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for key in ("last_modified", "last-modified", "Last-Modified"):
        value = field_value(get(source, key)) or field_value(data.get(key))
        if value:
            return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == "last-modified":
                    return field_value(value)
    return ""


def _parse(value: str) -> Any:
    parsed = parse_datetime(value)
    if parsed:
        return parsed
    try:
        return parsedate_to_datetime(value) if value else None
    except (TypeError, ValueError):
        return None
