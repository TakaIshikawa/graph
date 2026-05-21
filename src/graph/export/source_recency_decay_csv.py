"""CSV export for source-level recency decay bands."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from statistics import median
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source_project", "unit_count", "newest_date", "oldest_date", "median_age_days", "stale_unit_count", "fresh_unit_count", "decay_band"]
_DATE_KEYS = ("date", "source_date", "published_at", "published_date", "created_at", "updated_at", "ingested_at", "observed_at", "last_seen")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_recency_decay_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    reference_date: date | str | None = None,
    fresh_days: int = 30,
    stale_days: int = 365,
) -> str | dict[str, Any]:
    """Return or write source-level recency distribution bands."""
    unit_list = list(units)
    rows = _decay_rows(unit_list, _date_value(reference_date) or date.today(), fresh_days, stale_days)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _decay_rows(units: list[KnowledgeUnit | Mapping[str, Any]], reference_date: date, fresh_days: int, stale_days: int) -> list[dict[str, str | int]]:
    groups: dict[str, list[date]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for unit in units:
        source = _field_value(_get(unit, "source_project")) or "Unknown"
        counts[source] += 1
        unit_date = _unit_date(unit)
        if unit_date is not None:
            groups[source].append(unit_date)

    rows: list[dict[str, str | int]] = []
    for source in counts:
        dates = sorted(groups[source])
        ages = [(reference_date - value).days for value in dates]
        stale_count = sum(1 for age in ages if age > stale_days)
        fresh_count = sum(1 for age in ages if age <= fresh_days)
        median_age = int(median(ages)) if ages else ""
        band = "no_dates" if not dates else "fresh" if fresh_count == len(dates) else "stale" if stale_count == len(dates) else "mixed"
        rows.append(
            {
                "source_project": source,
                "unit_count": counts[source],
                "newest_date": dates[-1].isoformat() if dates else "",
                "oldest_date": dates[0].isoformat() if dates else "",
                "median_age_days": median_age,
                "stale_unit_count": stale_count,
                "fresh_unit_count": fresh_count,
                "decay_band": band,
            }
        )
    return sorted(rows, key=lambda row: _sort_key(row["source_project"]))


def _unit_date(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    metadata = _metadata(unit)
    for key in _DATE_KEYS:
        parsed = _date_value(_casefold_get(metadata, key)) or _date_value(_get(unit, key))
        if parsed is not None:
            return parsed
    return None


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
        return None


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


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
