"""Summarize source link rot risk signals."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, sort_key, source_id


def summarize_source_link_rot_risks(
    sources: Iterable[Mapping[str, Any] | object], now: datetime | None = None, sample_limit: int = 5
) -> dict[str, Any]:
    source_list = list(sources)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    rows = [_row(source, index, current) for index, source in enumerate(source_list)]
    counts = Counter(row["risk_bucket"] for row in rows)
    high = [row for row in rows if row["risk_bucket"] in {"client_error", "server_error", "fetch_error"}]
    archived = [row for row in rows if row["archived"]]
    stale = [row for row in rows if row["risk_bucket"] == "stale_check"]
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "risk_counts": dict(sorted(counts.items())),
        "high_risk_count": len(high),
        "archived_count": len(archived),
        "stale_check_count": len(stale),
        "samples": sorted((row for row in rows if row["risk_bucket"] != "ok"), key=lambda row: sort_key(row["source_id"]))[:limit],
    }


def _row(source: Mapping[str, Any] | object, index: int, now: datetime) -> dict[str, Any]:
    data = metadata(source)
    status = _int(_first(source, data, "status_code"))
    fetch_error = _first(source, data, "fetch_error")
    last_checked = _first(source, data, "last_checked_at")
    archived_url = _first(source, data, "archived_url") or _first(source, data, "archive_url")
    bucket = "ok"
    invalid_date = False
    if fetch_error:
        bucket = "fetch_error"
    elif status and 400 <= status < 500:
        bucket = "client_error"
    elif status and 500 <= status < 600:
        bucket = "server_error"
    else:
        parsed = parse_datetime(last_checked)
        invalid_date = bool(last_checked and parsed is None)
        if parsed is None and last_checked:
            bucket = "stale_check"
        elif parsed and (now - parsed).days > 90:
            bucket = "stale_check"
    return {
        "source_id": source_id(source) or str(index),
        "risk_bucket": bucket,
        "status_code": field_value(status or ""),
        "fetch_error": fetch_error,
        "last_checked_at": last_checked,
        "invalid_date": invalid_date,
        "archived": bool(archived_url),
        "url": _first(source, data, "url"),
    }


def _first(source: Mapping[str, Any] | object, data: Mapping[str, Any], key: str) -> str:
    return field_value(get(source, key)) or field_value(data.get(key))


def _int(value: object) -> int:
    try:
        return int(field_value(value))
    except ValueError:
        return 0
