"""CSV export for units with stale or missing content dates."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "title", "source", "last_seen_date", "age_days", "tags", "stale_reason"]
_DATE_KEYS = ("last_seen", "last_seen_at", "observed_at", "observed_date", "updated_at", "source_date", "published_at", "date", "created_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_stale_content_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    cutoff_date: date | datetime | str | None = None,
    max_age_days: int | None = 365,
    reference_date: date | datetime | str | None = None,
) -> str | dict[str, Any]:
    """Return or write units older than the cutoff, plus missing or malformed dates."""
    if max_age_days is not None and (not isinstance(max_age_days, int) or isinstance(max_age_days, bool) or max_age_days < 0):
        raise ValueError("max_age_days must be a non-negative integer or None")
    cutoff = _date_value(cutoff_date) if cutoff_date is not None else None
    if cutoff_date is not None and cutoff is None:
        raise ValueError("cutoff_date must be a date, datetime, or ISO date string")
    ref_date = _reference_date(reference_date)
    if cutoff is None and max_age_days is not None:
        cutoff = date.fromordinal(ref_date.toordinal() - max_age_days)

    unit_list = list(units)
    rows = _stale_rows(unit_list, cutoff_date=cutoff, reference_date=ref_date)
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
        "cutoff_date": cutoff.isoformat() if cutoff is not None else "",
        "reference_date": ref_date.isoformat(),
        "bytes_written": output_path.stat().st_size,
    }


def _stale_rows(
    units: list[KnowledgeUnit | Mapping[str, Any]],
    *,
    cutoff_date: date | None,
    reference_date: date,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        last_seen, had_malformed = _unit_date(unit)
        age_days = (reference_date - last_seen).days if last_seen is not None else None
        stale_reason = _stale_reason(last_seen, cutoff_date, had_malformed)
        if not stale_reason:
            continue
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
                "source": _field_value(_get(unit, "source_project")) or "Unknown",
                "last_seen_date": last_seen.isoformat() if last_seen is not None else "",
                "age_days": age_days if age_days is not None else "",
                "tags": _joined_unique(_unit_tags(unit)),
                "stale_reason": stale_reason,
            }
        )
    return sorted(rows, key=lambda row: (_reason_sort(row["stale_reason"]), -(int(row["age_days"]) if row["age_days"] != "" else -1), _sort_key(row["unit_id"])))


def _stale_reason(last_seen: date | None, cutoff_date: date | None, had_malformed: bool) -> str:
    if last_seen is None:
        return "malformed_date" if had_malformed else "missing_date"
    if cutoff_date is not None and last_seen < cutoff_date:
        return f"older_than_cutoff:{cutoff_date.isoformat()}"
    return ""


def _unit_date(unit: KnowledgeUnit | Mapping[str, Any]) -> tuple[date | None, bool]:
    metadata = _metadata(unit)
    had_malformed = False
    for source in (metadata, unit):
        for key in _DATE_KEYS:
            marker = object()
            raw = source.get(key, marker) if isinstance(source, Mapping) else getattr(source, key, marker)
            if raw is marker:
                continue
            parsed = _date_value(raw)
            if parsed is not None:
                return parsed, had_malformed
            if _field_value(raw):
                had_malformed = True
    return None, had_malformed


def _reference_date(value: date | datetime | str | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    parsed = _date_value(value)
    if parsed is None:
        raise ValueError("reference_date must be a date, datetime, or ISO date string")
    return parsed


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _field_value(value)
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


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    tags = _get(unit, "tags", [])
    if isinstance(tags, str):
        return [tags] if _field_value(tags) else []
    if isinstance(tags, Iterable):
        return [_field_value(tag) for tag in tags if _field_value(tag)]
    return []


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined_unique(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


def _reason_sort(reason: object) -> int:
    text = _field_value(reason)
    if text.startswith("older_than_cutoff"):
        return 0
    if text == "malformed_date":
        return 1
    if text == "missing_date":
        return 2
    return 3


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
