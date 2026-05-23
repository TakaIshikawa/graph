"""CSV export for unit metadata timestamp coverage."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "title", "timestamp_keys_present", "parsed_timestamp_count", "earliest_timestamp", "latest_timestamp", "precision_summary"]
_TIMESTAMP_KEY_RE = re.compile(r"(?:created|updated|published|modified|observed|timestamp|date|time|at)$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_metadata_timestamp_coverage_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-unit parseability coverage for timestamp metadata."""
    unit_list = list(units)
    rows = _coverage_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _coverage_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        metadata = _metadata(unit)
        present: list[str] = []
        parsed: list[datetime] = []
        precisions: dict[str, int] = {}
        invalid = 0
        for key, value in metadata.items():
            key_text = _field_value(key)
            if not key_text or not _TIMESTAMP_KEY_RE.search(key_text):
                continue
            present.append(key_text)
            for item in _flatten(value):
                timestamp, precision = _timestamp_value(item)
                if timestamp is None:
                    if _field_value(item):
                        invalid += 1
                    continue
                parsed.append(timestamp)
                precisions[precision] = precisions.get(precision, 0) + 1
        if invalid:
            precisions["invalid"] = invalid
        earliest = min(parsed, key=_timestamp_sort_key) if parsed else None
        latest = max(parsed, key=_timestamp_sort_key) if parsed else None
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
                "timestamp_keys_present": "; ".join(sorted(set(present), key=_sort_key)),
                "parsed_timestamp_count": len(parsed),
                "earliest_timestamp": earliest.isoformat() if earliest else "",
                "latest_timestamp": latest.isoformat() if latest else "",
                "precision_summary": _precision_summary(precisions),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["title"])))


def _timestamp_value(value: object) -> tuple[datetime | None, str]:
    if isinstance(value, datetime):
        return value, "datetime"
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()), "date"
    text = _field_value(value)
    if not text:
        return None, ""
    candidate = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
        precision = "date" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else "datetime"
        return parsed, precision
    except ValueError:
        return None, ""


def _precision_summary(precisions: dict[str, int]) -> str:
    return "; ".join(f"{key}:{precisions[key]}" for key in sorted(precisions, key=_sort_key))


def _timestamp_sort_key(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _flatten(value: object) -> list[object]:
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _flatten(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _flatten(child)]
    return [value]


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
