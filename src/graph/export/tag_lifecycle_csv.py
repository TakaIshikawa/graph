"""CSV export for tag lifecycle summaries."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["tag", "unit_count", "first_seen", "last_seen", "active_span_days", "undated_unit_count"]
_DATE_KEYS = ("observed_at", "observed_date", "source_date", "date", "published_at", "created_at", "updated_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_tag_lifecycle_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write lifecycle date summaries for tags."""
    unit_list = list(units)
    rows = _lifecycle_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _lifecycle_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"unit_ids": set(), "dates": [], "undated": 0})
    for unit in units:
        unit_date = _best_date(unit)
        for tag in _tags(unit):
            group = groups[tag]
            group["unit_ids"].add(_unit_id(unit))
            if unit_date is None:
                group["undated"] += 1
            else:
                group["dates"].append(unit_date)
    rows: list[dict[str, str | int]] = []
    for tag, group in groups.items():
        dates = group["dates"]
        first_seen = min(dates) if dates else None
        last_seen = max(dates) if dates else None
        rows.append(
            {
                "tag": tag,
                "unit_count": len(group["unit_ids"]),
                "first_seen": first_seen.isoformat() if first_seen else "",
                "last_seen": last_seen.isoformat() if last_seen else "",
                "active_span_days": (last_seen - first_seen).days if first_seen and last_seen else "",
                "undated_unit_count": group["undated"],
            }
        )
    return sorted(rows, key=lambda row: _sort_key(row["tag"]))


def _best_date(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    metadata = _metadata(unit)
    dates = [_date_value(_casefold_get(metadata, key)) for key in _DATE_KEYS]
    dates.extend(_date_value(_get(unit, key)) for key in ("created_at", "ingested_at", "updated_at"))
    parsed = [item for item in dates if item is not None]
    return min(parsed) if parsed else None


def _tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    value = _get(unit, "tags")
    return sorted({_field_value(item) for item in value if _field_value(item)}, key=_sort_key) if isinstance(value, list | tuple | set) else []


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
