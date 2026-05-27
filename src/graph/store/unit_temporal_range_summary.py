"""Temporal range summary for date-like unit metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

DATE_KEYS = ("created_at", "updated_at", "published_at", "date", "captured_at", "start_date", "end_date")


def summarize_unit_temporal_ranges(units: Iterable[Any]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    total = 0
    for index, unit in enumerate(units):
        total += 1
        unit_id = _unit_id(unit, index)
        meta = _metadata(unit)
        for key in DATE_KEYS:
            if key not in meta and _get(unit, key) in (None, ""):
                continue
            raw = _get(unit, key)
            if raw in (None, ""):
                raw = meta.get(key)
            row = rows.setdefault(key, {"key": key, "parsed": [], "invalid_count": 0})
            parsed = _parse(raw)
            if parsed is None:
                row["invalid_count"] += 1
            else:
                row["parsed"].append((parsed, unit_id, str(raw)))
    output = []
    for key in sorted(rows, key=_sort_key):
        parsed = sorted(rows[key]["parsed"], key=lambda item: (item[0], _sort_key(item[1])))
        earliest = parsed[0] if parsed else None
        latest = parsed[-1] if parsed else None
        output.append(
            {
                "key": key,
                "parsed_count": len(parsed),
                "invalid_count": rows[key]["invalid_count"],
                "earliest_value": earliest[0].isoformat() if earliest else "",
                "earliest_unit_id": earliest[1] if earliest else "",
                "latest_value": latest[0].isoformat() if latest else "",
                "latest_unit_id": latest[1] if latest else "",
                "span_days": (latest[0] - earliest[0]).days if earliest and latest else 0,
            }
        )
    return {"total_units": total, "rows": output}


def _parse(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _unit_id(unit: Any, index: int) -> str:
    return _text(_get(unit, "id") or _metadata(unit).get("id")) or str(index)


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
