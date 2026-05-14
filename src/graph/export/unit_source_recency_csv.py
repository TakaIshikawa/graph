"""CSV export for per-unit source recency."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "source_project",
    "source_entity_type",
    "title",
    "best_date",
    "age_days",
    "recency_bucket",
]
_DEFAULT_BUCKET_DAYS = (7, 30, 90, 365)
_DATE_KEY_RE = re.compile(r"(date|time|year|_at$)", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_source_recency_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    now: datetime | date | None = None,
    bucket_days: Sequence[int] = _DEFAULT_BUCKET_DAYS,
) -> str | dict[str, Any]:
    """Return or write one deterministic source recency row per unit."""
    buckets = _validate_bucket_days(bucket_days)
    unit_list = list(units)
    as_of = _as_datetime(now) if now is not None else datetime.now(timezone.utc)
    rows = _recency_rows(unit_list, now=as_of, bucket_days=buckets)
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
        "bucket_days": ",".join(str(day) for day in buckets),
        "bytes_written": output_path.stat().st_size,
    }


def _recency_rows(
    units: list[KnowledgeUnit],
    *,
    now: datetime,
    bucket_days: tuple[int, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        best_date = _best_date(unit)
        age_days = _age_days(best_date, now) if best_date is not None else None
        rows.append(
            {
                "unit_id": _field_value(unit.id),
                "source_project": _unit_source(unit),
                "source_entity_type": _unit_source_type(unit),
                "title": _field_value(unit.title),
                "best_date": _date_text(best_date),
                "age_days": str(age_days) if age_days is not None else "",
                "recency_bucket": _recency_bucket(age_days, bucket_days),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["source_project"]),
            _sort_key(row["source_entity_type"]),
            row["best_date"],
            _sort_key(row["unit_id"]),
        ),
    )


def _best_date(unit: KnowledgeUnit) -> datetime | date | None:
    for value in (getattr(unit, "updated_at", None), getattr(unit, "created_at", None)):
        parsed = _date_value(value)
        if parsed is not None:
            return parsed

    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    for key in sorted(metadata, key=_sort_key):
        if not _DATE_KEY_RE.search(_inline_text(key)):
            continue
        for value in _iter_values(metadata[key]):
            parsed = _date_value(value)
            if parsed is not None:
                return parsed
    return None


def _date_value(value: object) -> datetime | date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value

    text = _inline_text(value)
    if not text:
        return None
    if re.fullmatch(r"\d{4}", text):
        return date(int(text), 1, 1)
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = text.split("-")
        return date(int(year), int(month), 1)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _iter_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return sorted(value, key=_sort_key)
    if value is None:
        return []
    return [value]


def _age_days(value: datetime | date, now: datetime) -> int:
    if isinstance(value, datetime):
        compare = _as_datetime(value)
    else:
        compare = datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    return max(0, (now - compare).days)


def _as_datetime(value: datetime | date) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)


def _date_text(value: datetime | date | None) -> str:
    if value is None:
        return ""
    return value.isoformat()


def _recency_bucket(age_days: int | None, bucket_days: tuple[int, ...]) -> str:
    if age_days is None:
        return "undated"
    for day in bucket_days:
        if age_days <= day:
            return f"<= {day} days"
    return f"> {bucket_days[-1]} days"


def _validate_bucket_days(bucket_days: Sequence[int]) -> tuple[int, ...]:
    buckets = tuple(bucket_days)
    if (
        not buckets
        or any(not isinstance(day, int) or isinstance(day, bool) or day < 0 for day in buckets)
        or tuple(sorted(set(buckets))) != buckets
    ):
        raise ValueError("bucket_days must be a non-empty ascending sequence of unique non-negative integers")
    return buckets


def _render_csv(rows: list[dict[str, str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_source_type(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_entity_type) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
