"""Summarize saved query usage by type or source."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

_TYPE_KEYS = ("query_type", "type")
_SOURCE_KEYS = ("source", "source_project")
_RUN_COUNT_KEYS = ("run_count", "runs")
_LAST_RUN_KEYS = ("last_run_at", "last_executed_at", "ran_at")
_RESULT_COUNT_KEYS = ("result_count", "last_result_count", "results_count")
_ZERO_RUN_KEYS = ("zero_result_runs",)


def saved_query_usage_summary(
    saved_queries: Iterable[Any], *, reference_date: Any = None, stale_after_days: int = 30
) -> list[dict[str, Any]]:
    reference = _parse_dt(reference_date) if reference_date is not None else datetime.now(timezone.utc)
    groups: dict[str, dict[str, Any]] = {}
    for query in saved_queries:
        metadata = _metadata(query)
        group_key = _string(_first(query, metadata, _TYPE_KEYS) or _first(query, metadata, _SOURCE_KEYS)) or "unknown"
        group = groups.setdefault(
            group_key,
            {"query_type": group_key, "run_count": 0, "result_counts": [], "zero_result_runs": 0, "last_run_at": None},
        )
        run_count = _int(_first(query, metadata, _RUN_COUNT_KEYS), default=1)
        result_count = _int(_first(query, metadata, _RESULT_COUNT_KEYS), default=None)
        last_run = _parse_dt(_first(query, metadata, _LAST_RUN_KEYS))
        group["run_count"] += run_count
        if result_count is not None:
            group["result_counts"].append(result_count)
            if result_count == 0:
                group["zero_result_runs"] += run_count
        group["zero_result_runs"] += _int(_first(query, metadata, _ZERO_RUN_KEYS), default=0)
        if last_run and (group["last_run_at"] is None or last_run > group["last_run_at"]):
            group["last_run_at"] = last_run

    rows = []
    for group in groups.values():
        last_run = group["last_run_at"]
        rows.append(
            {
                "query_type": group["query_type"],
                "run_count": group["run_count"],
                "last_run_at": last_run.isoformat() if last_run else None,
                "average_result_count": round(sum(group["result_counts"]) / len(group["result_counts"]), 2)
                if group["result_counts"]
                else None,
                "zero_result_runs": group["zero_result_runs"],
                "stale_query": last_run is None or (reference - last_run).days > stale_after_days,
            }
        )
    return sorted(rows, key=lambda row: (-row["run_count"], row["query_type"]))


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _first(item: Any, metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _get(item, key)
        if value not in (None, ""):
            return value
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _int(value: Any, *, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
