"""CSV export for relation evidence age."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, render_csv, sort_key, write_csv

_FIELDNAMES = ["relation_id", "source_id", "target_id", "relation_type", "dated_evidence_count", "oldest_evidence_date", "newest_evidence_date", "age_bucket"]
_SOURCE_KEYS = ("source_id", "from_unit_id", "source_unit_id", "from_id", "source")
_TARGET_KEYS = ("target_id", "to_unit_id", "target_unit_id", "to_id", "target")
_RELATION_KEYS = ("relation_type", "relation", "type", "predicate")
_DATE_KEYS = ("evidence_date", "evidence_at", "date", "observed_at", "observed_date", "occurred_at", "timestamp", "published_at", "unit_date")
_EVIDENCE_KEYS = ("evidence", "evidence_items", "evidence_dates", "observations", "citations")


def export_relation_evidence_age_csv(relations: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    relation_list = list(relations)
    today = datetime.now(timezone.utc).date()
    rows = [_row(relation, index, today) for index, relation in enumerate(relation_list)]
    rows.sort(key=lambda row: (sort_key(row["age_bucket"]), sort_key(row["relation_type"]), sort_key(row["relation_id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "relation_count": len(relation_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(relation: Mapping[str, Any] | object, index: int, today: date) -> dict[str, str | int]:
    dates = sorted(_dates(relation))
    newest = dates[-1] if dates else None
    return {
        "relation_id": edge_id(relation) or str(index),
        "source_id": _value(relation, _SOURCE_KEYS),
        "target_id": _value(relation, _TARGET_KEYS),
        "relation_type": _value(relation, _RELATION_KEYS) or "unknown",
        "dated_evidence_count": len(dates),
        "oldest_evidence_date": dates[0].isoformat() if dates else "",
        "newest_evidence_date": newest.isoformat() if newest else "",
        "age_bucket": _age_bucket(newest, today),
    }


def _dates(relation: Any) -> list[date]:
    values: list[Any] = []
    values.extend(_values_for_keys(relation, _DATE_KEYS))
    meta = metadata(relation)
    values.extend(_values_for_keys(meta, _DATE_KEYS))
    values.extend(_evidence_values(relation))
    values.extend(_evidence_values(meta))
    return [parsed for value in values if (parsed := _date_value(value))]


def _evidence_values(container: Any) -> list[Any]:
    values: list[Any] = []
    for key in _EVIDENCE_KEYS:
        raw = get(container, key) if not isinstance(container, Mapping) else container.get(key)
        items = raw if isinstance(raw, list | tuple | set) else [raw]
        for item in items:
            if isinstance(item, Mapping):
                values.extend(_values_for_keys(item, _DATE_KEYS))
                if isinstance(item.get("metadata"), Mapping):
                    values.extend(_values_for_keys(item["metadata"], _DATE_KEYS))
            elif item is not None:
                values.append(item)
    return values


def _values_for_keys(container: Any, keys: tuple[str, ...]) -> list[Any]:
    return [value for key in keys if (value := (container.get(key) if isinstance(container, Mapping) else get(container, key))) is not None]


def _date_value(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = field_value(value)
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


def _age_bucket(value: date | None, today: date) -> str:
    if value is None:
        return "unknown"
    days = (today - value).days
    if days <= 30:
        return "0-30d"
    if days <= 90:
        return "31-90d"
    if days <= 365:
        return "91-365d"
    return "365d+"


def _value(item: Any, keys: tuple[str, ...]) -> str:
    meta = metadata(item)
    for key in keys:
        text = field_value(get(item, key)) or field_value(meta.get(key))
        if text:
            return text
    return ""
