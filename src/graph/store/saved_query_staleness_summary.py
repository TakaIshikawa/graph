"""Evaluate saved query result freshness against graph updates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any


def saved_query_staleness_summary(
    saved_queries: Iterable[Mapping[str, Any]],
    units: Iterable[Any],
    *,
    now: str | datetime | None = None,
    max_age: timedelta | None = None,
) -> list[dict[str, Any]]:
    current = _parse_dt(now) if now is not None else datetime.now(timezone.utc)
    threshold = max_age
    newest_by_source = _newest_source_updates(units)
    newest_any = max((value for value in newest_by_source.values() if value), default=None)
    rows = []
    for query in saved_queries:
        last_run_value = query.get("last_run_at")
        last_run = _parse_dt(last_run_value)
        filters = query.get("filters") if isinstance(query.get("filters"), Mapping) else {}
        source_filter = filters.get("source_project") or filters.get("source")
        newest = newest_by_source.get(str(source_filter)) if source_filter else newest_any
        reasons = []
        if last_run_value in (None, ""):
            reasons.append("never_run")
        if newest and (last_run is None or newest > last_run):
            reasons.append("graph_updated")
        if threshold is not None and last_run is not None and current - last_run > threshold:
            reasons.append("max_age_exceeded")
        rows.append(
            {
                "name": str(query.get("name")),
                "stale": bool(reasons),
                "newest_relevant_update_at": newest.isoformat() if newest else None,
                "refresh_reasons": reasons,
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def _newest_source_updates(units: Iterable[Any]) -> dict[str, datetime | None]:
    newest: dict[str, datetime | None] = {}
    for unit in units:
        source = str(_get(unit, "source_project"))
        timestamp = _parse_dt(_get(unit, "updated_at") or _get(unit, "ingested_at"))
        if timestamp and (source not in newest or newest[source] is None or timestamp > newest[source]):
            newest[source] = timestamp
    return newest


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)
