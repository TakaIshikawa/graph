"""CSV export for stale knowledge unit review."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "last_activity_date",
    "age_days",
    "staleness_bucket",
    "has_recent_relations",
    "relation_count",
    "tags",
]
_UNKNOWN = "Unknown"
_OBSERVED_KEYS = ("observed_at", "observed_date", "last_observed_at", "last_seen")
_UPDATED_KEYS = ("updated_at", "updated_date", "modified_at")
_CREATED_KEYS = ("created_at", "created_date")
_SOURCE_DATE_KEYS = ("source_date", "published_at", "published_date", "date")
_EDGE_DATE_KEYS = ("observed_at", "observed_date", "updated_at", "created_at", "date", "source_date")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_staleness_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]] = (),
    path: str | Path | None = None,
    *,
    reference_date: date | datetime | str | None = None,
    recent_relation_days: int = 90,
) -> str | dict[str, Any]:
    """Return or write per-unit staleness rows using an injectable reference date."""
    if not isinstance(recent_relation_days, int) or isinstance(recent_relation_days, bool) or recent_relation_days < 0:
        raise ValueError("recent_relation_days must be a non-negative integer")

    unit_list = list(units)
    edge_list = list(edges)
    ref_date = _reference_date(reference_date)
    rows = _staleness_rows(unit_list, edge_list, reference_date=ref_date, recent_relation_days=recent_relation_days)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "edge_count": len(edge_list),
        "rows_exported": len(rows),
        "reference_date": ref_date.isoformat(),
        "bytes_written": output_path.stat().st_size,
    }


def _staleness_rows(
    units: list[KnowledgeUnit | Mapping[str, Any]],
    edges: list[KnowledgeEdge | Mapping[str, Any]],
    *,
    reference_date: date,
    recent_relation_days: int,
) -> list[dict[str, str | int]]:
    relation_counts, recent_units = _relation_context(edges, reference_date=reference_date, recent_relation_days=recent_relation_days)
    rows: list[dict[str, str | int]] = []
    for unit in units:
        unit_id = _field_value(_get(unit, "id"))
        last_activity_date = _unit_activity_date(unit)
        age_days = (reference_date - last_activity_date).days if last_activity_date is not None else None
        rows.append(
            {
                "unit_id": unit_id,
                "title": _field_value(_get(unit, "title")),
                "source_project": _field_value(_get(unit, "source_project")) or _UNKNOWN,
                "last_activity_date": last_activity_date.isoformat() if last_activity_date is not None else "",
                "age_days": age_days if age_days is not None else "",
                "staleness_bucket": _staleness_bucket(age_days),
                "has_recent_relations": "true" if unit_id in recent_units else "false",
                "relation_count": relation_counts[unit_id],
                "tags": _joined_unique(_unit_tags(unit)),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _bucket_sort(row["staleness_bucket"]),
            -int(row["age_days"] or -1),
            _sort_key(row["source_project"]),
            _sort_key(row["unit_id"]),
        ),
    )


def _relation_context(
    edges: list[KnowledgeEdge | Mapping[str, Any]],
    *,
    reference_date: date,
    recent_relation_days: int,
) -> tuple[Counter[str], set[str]]:
    relation_counts: Counter[str] = Counter()
    recent_units: set[str] = set()
    for edge in edges:
        endpoints = [_field_value(_get(edge, "from_unit_id")), _field_value(_get(edge, "to_unit_id"))]
        endpoints = [endpoint for endpoint in endpoints if endpoint]
        relation_counts.update(endpoints)
        edge_date = _edge_date(edge)
        if edge_date is None:
            continue
        age_days = (reference_date - edge_date).days
        if 0 <= age_days <= recent_relation_days:
            recent_units.update(endpoints)
    return relation_counts, recent_units


def _unit_activity_date(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    metadata = _metadata(unit)
    for keys in (_OBSERVED_KEYS, _UPDATED_KEYS, _CREATED_KEYS, _SOURCE_DATE_KEYS):
        for key in keys:
            parsed = _date_value(metadata.get(key))
            if parsed is not None:
                return parsed
    for keys in (_OBSERVED_KEYS, _UPDATED_KEYS, _CREATED_KEYS, _SOURCE_DATE_KEYS):
        for key in keys:
            parsed = _date_value(_get(unit, key))
            if parsed is not None:
                return parsed
    return None


def _edge_date(edge: KnowledgeEdge | Mapping[str, Any]) -> date | None:
    metadata = _metadata(edge)
    for key in _EDGE_DATE_KEYS:
        parsed = _date_value(metadata.get(key))
        if parsed is not None:
            return parsed
    for key in _EDGE_DATE_KEYS:
        parsed = _date_value(_get(edge, key))
        if parsed is not None:
            return parsed
    return None


def _reference_date(value: date | datetime | str | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    parsed = _date_value(value)
    if parsed is None:
        raise ValueError("reference_date must be a date, datetime, or ISO date string")
    return parsed


def _staleness_bucket(age_days: int | None) -> str:
    if age_days is None:
        return "missing_date"
    if age_days <= 30:
        return "current"
    if age_days <= 90:
        return "aging"
    if age_days <= 365:
        return "stale"
    return "dormant"


def _bucket_sort(bucket: object) -> int:
    order = {"dormant": 0, "stale": 1, "aging": 2, "current": 3, "missing_date": 4}
    return order.get(_field_value(bucket), 5)


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    tags = _get(unit, "tags", [])
    if isinstance(tags, str):
        return [tags] if _field_value(tags) else []
    if isinstance(tags, Iterable):
        return [_field_value(tag) for tag in tags if _field_value(tag)]
    return []


def _metadata(value: KnowledgeUnit | KnowledgeEdge | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(value, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


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


def _joined_unique(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


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
