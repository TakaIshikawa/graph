"""Summarize unit review queue signals by source."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

_STATUS_KEYS = ("review_status", "status", "triage_status")
_DUE_KEYS = ("review_due", "due_at", "next_review_at")
_PRIORITY_KEYS = ("priority", "review_priority")
_REVIEW_REQUESTED_STATUSES = {"review", "review_requested", "needs_review", "pending_review", "triage", "requested"}
_BLOCKED_STATUSES = {"blocked", "blocker", "on_hold", "on hold", "stuck"}
_HIGH_PRIORITY_TEXT = {"high", "highest", "urgent", "critical", "p0", "p1", "1"}


def summarize_unit_review_queue(units: Iterable[Any], *, now: str | datetime | None = None) -> dict[str, Any]:
    """Group units by source and count actionable review metadata."""

    reference = _parse_dt(now) if now is not None else datetime.now(timezone.utc)
    grouped: dict[str, list[Any]] = defaultdict(list)
    total_units = 0
    for unit in units:
        total_units += 1
        grouped[_source(unit)].append(unit)

    rows = [_row(source, grouped[source], reference) for source in sorted(grouped, key=_sort_key)]
    return {"total_units": total_units, "rows": rows, "source_summaries": rows}


def _row(source: str, units: list[Any], reference: datetime) -> dict[str, Any]:
    due_units = []
    for unit in units:
        due_at = _due_at(unit)
        if due_at is not None:
            due_units.append((_unit_id(unit), due_at))

    due_units.sort(key=lambda item: (item[1], _sort_key(item[0])))
    return {
        "source": source,
        "unit_count": len(units),
        "review_requested_count": sum(1 for unit in units if _is_review_requested(unit)),
        "overdue_count": sum(1 for _unit_id, due_at in due_units if due_at < reference),
        "high_priority_count": sum(1 for unit in units if _is_high_priority(unit)),
        "blocked_count": sum(1 for unit in units if _is_blocked(unit)),
        "next_due_unit_id": due_units[0][0] if due_units else "",
    }


def _is_review_requested(unit: Any) -> bool:
    status = _normalize(_first(_metadata(unit), _STATUS_KEYS))
    return status in _REVIEW_REQUESTED_STATUSES or "review" in status


def _is_blocked(unit: Any) -> bool:
    return _normalize(_first(_metadata(unit), _STATUS_KEYS)) in _BLOCKED_STATUSES


def _is_high_priority(unit: Any) -> bool:
    priority = _first(_metadata(unit), _PRIORITY_KEYS)
    if isinstance(priority, bool):
        return False
    try:
        return float(priority) >= 8
    except (TypeError, ValueError):
        return _normalize(priority) in _HIGH_PRIORITY_TEXT


def _due_at(unit: Any) -> datetime | None:
    value = _first(_metadata(unit), _DUE_KEYS)
    if value in (None, ""):
        return None
    try:
        return _parse_dt(value)
    except ValueError:
        return None


def _source(unit: Any) -> str:
    meta = _metadata(unit)
    return _text(_get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"


def _unit_id(unit: Any) -> str:
    return _text(_get(unit, "id") or _get(unit, "unit_id"))


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_dt(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _normalize(value: Any) -> str:
    return _text(value).casefold()


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
