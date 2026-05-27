"""Summarize source/import error messages found in unit metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_FIELDS = {"error", "errors", "warning", "warnings", "exception", "exceptions", "status_message", "failure_reason"}
_UNKNOWN_SOURCE = "Unknown"


def summarize_source_error_messages(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    total_units = 0
    counts: Counter[tuple[str, str, str]] = Counter()
    examples: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for unit in units:
        total_units += 1
        source = _source(unit)
        for field, value in _walk(metadata(unit)):
            field_name = field.casefold()
            if field_name not in _FIELDS:
                continue
            for severity, message in _message_entries(field_name, value):
                key = (source, severity, message)
                counts[key] += 1
                if len(examples[key]) < sample_limit:
                    examples[key].append(unit_id(unit))

    rows = [
        {
            "source": source,
            "severity": severity,
            "message": message,
            "count": count,
            "sample_unit_ids": examples[(source, severity, message)],
        }
        for (source, severity, message), count in counts.items()
    ]
    rows.sort(key=lambda row: (sort_key(row["source"]), sort_key(row["severity"]), -row["count"], sort_key(row["message"])))
    return {"total_units": total_units, "message_count": sum(counts.values()), "messages": rows}


def _source(unit: Any) -> str:
    return field_value(get(unit, "source_project") or metadata(unit).get("source_project")) or _UNKNOWN_SOURCE


def _walk(value: object) -> list[tuple[str, object]]:
    found: list[tuple[str, object]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.append((field_value(key), child))
            found.extend(_walk(child))
    elif isinstance(value, list | tuple | set):
        for child in value:
            found.extend(_walk(child))
    return found


def _message_entries(field_name: str, value: object) -> list[tuple[str, str]]:
    return [(_severity(field_name, item), message) for item, message in _messages(value)]


def _messages(value: object) -> list[tuple[object, str]]:
    if isinstance(value, Mapping):
        message = field_value(value.get("message") or value.get("detail") or value.get("text") or value.get("reason"))
        return [(value, message)] if message else []
    if isinstance(value, list | tuple | set):
        return [message for child in value for message in _messages(child)]
    message = field_value(value)
    return [(value, message)] if message else []


def _severity(field_name: str, value: object) -> str:
    if isinstance(value, Mapping):
        explicit = field_value(value.get("severity") or value.get("level")).casefold()
        if explicit:
            return explicit
    if "warning" in field_name:
        return "warning"
    if "status" in field_name:
        return "info"
    return "error"
