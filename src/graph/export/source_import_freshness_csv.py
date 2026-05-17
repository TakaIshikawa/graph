"""CSV export for source import freshness by project."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "unit_count",
    "newest_updated_at",
    "oldest_updated_at",
    "days_since_newest",
    "days_since_oldest",
    "stale_unit_count",
    "freshness_bucket",
]
_STALE_AFTER_DAYS = 90
_AGING_AFTER_DAYS = 30
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_import_freshness_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    now: datetime | date | str | None = None,
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write import freshness grouped by source project."""
    unit_list = list(units)
    rows = _freshness_rows(unit_list, _date_value(now) or datetime.now(timezone.utc).date())
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_project_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _freshness_rows(
    units: list[KnowledgeUnit | Mapping[str, Any]],
    today: date,
) -> list[dict[str, str | int]]:
    groups: dict[str, list[date | None]] = defaultdict(list)
    for unit in units:
        groups[_unit_source(unit)].append(_unit_timestamp(unit))

    rows: list[dict[str, str | int]] = []
    for source_project in sorted(groups, key=_sort_key):
        values = groups[source_project]
        dated_values = sorted(value for value in values if value is not None)
        newest = dated_values[-1] if dated_values else None
        oldest = dated_values[0] if dated_values else None
        days_since_newest = _days_since(today, newest)
        days_since_oldest = _days_since(today, oldest)
        rows.append(
            {
                "source_project": source_project,
                "unit_count": len(values),
                "newest_updated_at": newest.isoformat() if newest else "",
                "oldest_updated_at": oldest.isoformat() if oldest else "",
                "days_since_newest": days_since_newest,
                "days_since_oldest": days_since_oldest,
                "stale_unit_count": sum(
                    1 for value in values if value is not None and _days_since(today, value) > _STALE_AFTER_DAYS
                ),
                "freshness_bucket": _freshness_bucket(days_since_newest),
            }
        )
    return rows


def _unit_timestamp(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    for key in ("updated_at", "ingested_at", "imported_at"):
        value = _date_value(_get(unit, key))
        if value is not None:
            return value
    metadata = _metadata(unit)
    for key in ("updated_at", "ingested_at", "imported_at", "last_imported_at", "last_sync_at"):
        value = _date_value(metadata.get(key))
        if value is not None:
            return value
    return None


def _freshness_bucket(days_since_newest: int | str) -> str:
    if days_since_newest == "":
        return "empty"
    days = int(days_since_newest)
    if days > _STALE_AFTER_DAYS:
        return "stale"
    if days > _AGING_AFTER_DAYS:
        return "aging"
    return "fresh"


def _days_since(today: date, value: date | None) -> int | str:
    if value is None:
        return ""
    return max((today - value).days, 0)


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


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _unit_source(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_project")) or "Unknown"


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


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
