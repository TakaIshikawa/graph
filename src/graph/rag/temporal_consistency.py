"""Check retrieved RAG evidence for temporal consistency with a query window."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

from graph.rag._analysis_utils import MISSING, coerce_now, parse_date, result_id, string, value

_DATE_FIELDS = ("published_at", "updated_at", "created_at", "date", "start_date", "end_date")
_WINDOW_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\s*(?:to|through|until|-)\s*(\d{4}-\d{2}-\d{2})\b", re.I)
_YEAR_RE = re.compile(r"\b(?:in|during|for)\s+(\d{4})\b", re.I)
_BUCKETS = ("in_window", "before_window", "after_window", "stale", "future", "missing_date")
_WARNINGS = (
    ("future", "future_dated_evidence"),
    ("missing_date", "missing_date_evidence"),
    ("out_of_window", "out_of_window_evidence"),
    ("stale", "stale_evidence"),
)


def analyze_temporal_consistency(
    query_text: str,
    results: Iterable[Any],
    *,
    reference_date: Any = None,
    window_start: Any = None,
    window_end: Any = None,
    stale_after_days: int = 365,
) -> dict[str, Any]:
    """Return deterministic date buckets and warnings for retrieved evidence."""
    if isinstance(stale_after_days, bool) or not isinstance(stale_after_days, int) or stale_after_days < 0:
        raise ValueError("stale_after_days must be a non-negative integer")

    query = " ".join(str(query_text or "").strip().split())
    today = coerce_now(reference_date)
    inferred_start, inferred_end = _query_window(query)
    start = parse_date(window_start) if window_start is not None else inferred_start
    end = parse_date(window_end) if window_end is not None else inferred_end
    if start is not None and end is not None and start > end:
        start, end = end, start

    rows = []
    bucket_ids = {bucket: [] for bucket in _BUCKETS}
    counts = {
        "total": 0,
        "dated": 0,
        "in_window": 0,
        "stale": 0,
        "future": 0,
        "missing_date": 0,
        "out_of_window": 0,
    }
    dates: list[date] = []

    for index, result in enumerate(results or []):
        counts["total"] += 1
        row = _temporal_row(result, index, today, start, end, stale_after_days)
        rows.append(row)
        for key in ("dated", "in_window", "stale", "future", "missing_date", "out_of_window"):
            if row[key]:
                counts[key] += 1
        bucket_ids[row["bucket"]].append(row["result_id"])
        if row["date"] is not None:
            dates.append(parse_date(row["date"]) or today)

    warnings = [warning for key, warning in _WARNINGS if counts[key]]
    return {
        "query": query,
        "reference_date": today.isoformat(),
        "window": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "source": _window_source(window_start, window_end, inferred_start, inferred_end),
        },
        "counts": counts,
        "date_buckets": {bucket: len(bucket_ids[bucket]) for bucket in _BUCKETS},
        "result_ids": {bucket: sorted(bucket_ids[bucket], key=_sort_key) for bucket in _BUCKETS},
        "oldest_date": min(dates).isoformat() if dates else None,
        "newest_date": max(dates).isoformat() if dates else None,
        "results": rows,
        "warnings": warnings,
    }


def _temporal_row(
    result: Any,
    index: int,
    today: date,
    window_start: date | None,
    window_end: date | None,
    stale_after_days: int,
) -> dict[str, Any]:
    result_id_ = result_id(result, index)
    start, end, date_field = _date_range(result)
    observed = end or start
    dated = observed is not None
    future = bool(start and start > today) or bool(end and end > today)
    stale = bool(observed and observed < today - timedelta(days=stale_after_days))
    has_window = window_start is not None or window_end is not None
    out_of_window = dated and has_window and not _overlaps(start or end, end or start, window_start, window_end)
    in_window = dated and not out_of_window and window_start is not None and window_end is not None
    bucket = _bucket(start, end, dated, stale, future, out_of_window, window_start, window_end)
    return {
        "result_id": result_id_,
        "source": string(value(result, "source")),
        "date": observed.isoformat() if observed else None,
        "start_date": start.isoformat() if start else None,
        "end_date": end.isoformat() if end else None,
        "date_field": date_field,
        "dated": dated,
        "in_window": in_window,
        "stale": stale,
        "future": future,
        "missing_date": not dated,
        "out_of_window": out_of_window,
        "bucket": bucket,
    }


def _date_range(result: Any) -> tuple[date | None, date | None, str | None]:
    parsed_by_field: dict[str, date] = {}
    for field in _DATE_FIELDS:
        parsed = parse_date(value(result, field))
        if parsed is not None:
            parsed_by_field[field] = parsed
    start = parsed_by_field.get("start_date")
    end = parsed_by_field.get("end_date")
    if start is None and end is None:
        for field in _DATE_FIELDS:
            if field in parsed_by_field:
                return parsed_by_field[field], parsed_by_field[field], field
        return None, None, None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is not None and end is not None and start > end:
        start, end = end, start
    return start, end, "start_date" if "start_date" in parsed_by_field else "end_date"


def _query_window(query: str) -> tuple[date | None, date | None]:
    match = _WINDOW_RE.search(query)
    if match:
        return parse_date(match.group(1)), parse_date(match.group(2))
    match = _YEAR_RE.search(query)
    if match:
        year = int(match.group(1))
        return date(year, 1, 1), date(year, 12, 31)
    return None, None


def _window_source(window_start: Any, window_end: Any, inferred_start: date | None, inferred_end: date | None) -> str | None:
    if window_start is not None or window_end is not None:
        return "parameters"
    if inferred_start is not None or inferred_end is not None:
        return "query"
    return None


def _overlaps(
    start: date | None,
    end: date | None,
    window_start: date | None,
    window_end: date | None,
) -> bool:
    if window_start is None and window_end is None:
        return False
    if start is None and end is None:
        return False
    start = start or end
    end = end or start
    if start is None or end is None:
        return False
    if window_start is not None and end < window_start:
        return False
    if window_end is not None and start > window_end:
        return False
    return True


def _bucket(
    start: date | None,
    end: date | None,
    dated: bool,
    stale: bool,
    future: bool,
    out_of_window: bool,
    window_start: date | None,
    window_end: date | None,
) -> str:
    if not dated:
        return "missing_date"
    if future:
        return "future"
    if out_of_window and window_start is not None and end is not None and end < window_start:
        return "before_window"
    if out_of_window and window_end is not None and start is not None and start > window_end:
        return "after_window"
    if window_start is not None or window_end is not None:
        return "in_window"
    if stale:
        return "stale"
    return "in_window"


def _sort_key(value_: object) -> tuple[str, str]:
    text = "" if value_ is MISSING or value_ is None else str(value_)
    return (text.casefold(), text)
