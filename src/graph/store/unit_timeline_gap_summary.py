"""Timeline gap summary for store units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

TIME_KEYS = ("created_at", "updated_at", "published_at", "date", "captured_at")


def summarize_unit_timeline_gaps(units: Iterable[Any], *, group_by: tuple[str, ...] = ("collection", "source")) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[tuple[datetime, str]]] = defaultdict(list)
    skipped = 0
    total = 0
    for index, unit in enumerate(units):
        total += 1
        ts = _timestamp(unit)
        if ts is None:
            skipped += 1
            continue
        groups[(_group(unit, group_by[0]), _group(unit, group_by[1]))].append((ts, _unit_id(unit, index)))

    rows = []
    for key in sorted(groups, key=lambda item: (_sort_key(item[0]), _sort_key(item[1]))):
        points = sorted(groups[key], key=lambda item: (item[0], _sort_key(item[1])))
        largest = (0, "", "")
        for before, after in zip(points, points[1:]):
            days = (after[0] - before[0]).days
            if (days, before[1], after[1]) > largest:
                largest = (days, before[1], after[1])
        rows.append(
            {
                "collection": key[0],
                "source": key[1],
                "unit_count": len(points),
                "first_timestamp": points[0][0].date().isoformat(),
                "last_timestamp": points[-1][0].date().isoformat(),
                "largest_gap_days": largest[0],
                "gap_days": largest[0],
                "before_unit_id": largest[1],
                "after_unit_id": largest[2],
            }
        )
    return {"total_units": total, "skipped_units": skipped, "rows": rows}


def _timestamp(unit: Any) -> datetime | None:
    meta = _metadata(unit)
    for key in TIME_KEYS:
        value = _get(unit, key)
        if value in (None, ""):
            value = meta.get(key)
        parsed = _parse(value)
        if parsed is not None:
            return parsed
    return None


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


def _group(unit: Any, key: str) -> str:
    meta = _metadata(unit)
    aliases = {
        "collection": ("collection", "collection_id", "collection_name"),
        "source": ("source", "source_project", "source_type"),
    }.get(key, (key,))
    for alias in aliases:
        text = _text(_get(unit, alias)) or _text(meta.get(alias))
        if text:
            return text
    return "unknown"


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _unit_id(unit: Any, index: int) -> str:
    return _text(_get(unit, "id") or _metadata(unit).get("id")) or str(index)


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(getattr(value, "value", value)).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
