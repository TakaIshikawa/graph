"""TimelineJS JSON export helpers for dated knowledge units."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, overload

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

START_DATE_METADATA_KEYS = (
    "start_date",
    "event_start",
    "date",
    "published_at",
    "completed_at",
    "created_at",
    "updated_at",
    "ingested_at",
)
END_DATE_METADATA_KEYS = ("end_date", "event_end")
URL_METADATA_KEYS = (
    "media.url",
    "image_url",
    "thumbnail_url",
    "url",
    "source_url",
    "external_url",
    "uri",
)


@overload
def export_units_to_timelinejs(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: None = None,
) -> str: ...


@overload
def export_units_to_timelinejs(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path,
) -> dict[str, Any]: ...


def export_units_to_timelinejs(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write units as TimelineJS-compatible JSON."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    events = []
    skipped_count = 0

    for unit in unit_list:
        event = _unit_event(unit)
        if event is None:
            skipped_count += 1
            continue
        events.append(event)

    events.sort(key=_event_sort_key)
    text = json.dumps({"events": [event for _, _, event in events]}, ensure_ascii=False, sort_keys=True, indent=2)
    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "event_count": len(events),
        "skipped_count": skipped_count,
        "bytes_written": output_path.stat().st_size,
    }


def _unit_event(unit: KnowledgeUnit) -> tuple[str, tuple[str, str, str], dict[str, Any]] | None:
    start = _unit_start_date(unit)
    if start is None:
        return None

    event: dict[str, Any] = {
        "start_date": _timeline_date(start),
        "text": {
            "headline": _clean_text(unit.title) or "Untitled graph unit",
            "text": _clean_text(unit.content),
        },
        "group": _scalar_text(unit.source_project),
        "tags": sorted(_clean_text(_scalar_text(tag)) for tag in unit.tags if _clean_text(_scalar_text(tag))),
    }

    end = _unit_end_date(unit)
    if end is not None:
        event["end_date"] = _timeline_date(end)

    media_url = _first_text(unit.metadata, URL_METADATA_KEYS)
    if media_url:
        event["media"] = {"url": media_url}

    return (_date_sort_value(start), _unit_sort_key(unit), event)


def _unit_start_date(unit: KnowledgeUnit) -> datetime | date | None:
    for key in START_DATE_METADATA_KEYS:
        parsed = _parse_date_value(_nested_value(unit.metadata, key))
        if parsed is not None:
            return parsed
    for value in (unit.created_at, unit.updated_at, unit.ingested_at):
        parsed = _parse_date_value(value)
        if parsed is not None:
            return parsed
    return None


def _unit_end_date(unit: KnowledgeUnit) -> datetime | date | None:
    for key in END_DATE_METADATA_KEYS:
        parsed = _parse_date_value(_nested_value(unit.metadata, key))
        if parsed is not None:
            return parsed
    return None


def _timeline_date(value: datetime | date) -> dict[str, int]:
    if isinstance(value, datetime):
        normalized = _aware_utc(value)
        result = {
            "year": normalized.year,
            "month": normalized.month,
            "day": normalized.day,
            "hour": normalized.hour,
            "minute": normalized.minute,
        }
        if normalized.second:
            result["second"] = normalized.second
        return result
    return {"year": value.year, "month": value.month, "day": value.day}


def _parse_date_value(value: Any) -> datetime | date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _aware_utc(value)
    if isinstance(value, date):
        return value
    text = _clean_text(_scalar_text(value))
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        try:
            return datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            return None
    if match := re.match(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$", text):
        year, month, day = match.groups()
        try:
            return date(int(year), int(month or 1), int(day or 1))
        except ValueError:
            return None
    try:
        if text.endswith("Z") and "-" not in text:
            return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if "T" in text and "-" not in text:
            return datetime.strptime(text, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return _aware_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        return None


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _date_sort_value(value: datetime | date) -> str:
    if isinstance(value, datetime):
        return _aware_utc(value).isoformat()
    return value.isoformat()


def _event_sort_key(event: tuple[str, tuple[str, str, str], dict[str, Any]]) -> tuple[str, tuple[str, str, str]]:
    return (event[0], event[1])


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (_clean_text(unit.title), str(unit.id or ""), str(unit.source_id or ""))


def _first_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _clean_text(_scalar_text(_nested_value(metadata, key)))
        if text:
            return text
    return ""


def _nested_value(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata.get(key)
    current: Any = metadata
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current.get(part)
    return current


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _scalar_text(value.model_dump())
    return str(value)


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())
