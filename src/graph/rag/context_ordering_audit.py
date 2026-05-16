"""Audit retrieved context ordering for answer synthesis."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()


def _payload(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) and result else result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text:
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first(result, ("id", "unit_id", "source_id")) or f"result-{index + 1}"


def _number(result: Any, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        for value in _candidate_values(result, key):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _date_value(result: Any) -> datetime | None:
    dates = [
        parsed
        for key in ("date", "published_at", "publication_date", "updated_at", "created_at")
        for value in _candidate_values(result, key)
        if (parsed := _parse_datetime(value)) is not None
    ]
    return max(dates) if dates else None


def _source(result: Any) -> str:
    raw = _first(result, ("source", "source_project", "domain", "url", "canonical_url")) or "unknown"
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    return (parsed.hostname or raw).casefold().removeprefix("www.")


def _normalize_now(now: date | datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if not isinstance(now, date | datetime):
        raise ValueError("now must be a date, datetime, or None")
    parsed = _parse_datetime(now)
    if parsed is None:
        raise ValueError("now must be a date, datetime, or None")
    return parsed


def audit_context_ordering(results: Iterable[Any], *, now: date | datetime | None = None) -> dict[str, Any]:
    """Return ordering risks for retrieved context rows."""
    _normalize_now(now)
    rows = [
        {
            "index": index,
            "result_id": _result_id(result, index),
            "score": _number(result, ("score", "relevance_score", "similarity")),
            "rank": _number(result, ("rank", "position")),
            "date": _date_value(result),
            "source": _source(result),
        }
        for index, result in enumerate(results)
    ]
    issues: list[dict[str, Any]] = []
    for i, earlier in enumerate(rows):
        score_reported = False
        date_reported = False
        for later in rows[i + 1 :]:
            if not score_reported and earlier["score"] is not None and later["score"] is not None and later["score"] - earlier["score"] >= 0.15:
                issues.append({"type": "low-score-before-high-score", "severity": "medium", "result_ids": [earlier["result_id"], later["result_id"]]})
                score_reported = True
            if not date_reported and earlier["date"] is not None and later["date"] is not None and (later["date"] - earlier["date"]).days >= 180:
                issues.append({"type": "stale-before-fresh", "severity": "medium", "result_ids": [earlier["result_id"], later["result_id"]]})
                date_reported = True
    top = rows[: min(3, len(rows))]
    if len(top) >= 3:
        source, count = Counter(row["source"] for row in top).most_common(1)[0]
        if count == len(top) and source != "unknown":
            issues.append({"type": "same-source-top-block", "severity": "low", "result_ids": [row["result_id"] for row in top], "source": source})

    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda row: (severity_order[row["severity"]], row["type"], row["result_ids"]))
    return {"issue_count": len(issues), "issues": issues, "summary": {"result_count": len(rows), "issue_count": len(issues)}}
