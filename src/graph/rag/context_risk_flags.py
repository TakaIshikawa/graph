"""Flag retrieval context risks for RAG/search result sets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


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
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING:
            yield value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            value = unit_metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).split())
    return text or None


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first(result, ("id", "unit_id", "source_id")) or f"result-{index + 1}"


def _normalize_now(now: date | datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, datetime):
        return now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    if isinstance(now, date):
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    raise ValueError("now must be a date, datetime, or None")


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
            try:
                parsed_date = date.fromisoformat(value.strip())
            except ValueError:
                return None
            parsed = datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=timezone.utc)
    else:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _dates(result: Any) -> list[datetime]:
    return [
        parsed
        for key in ("published_at", "publication_date", "updated_at", "created_at", "date")
        for value in _candidate_values(result, key)
        if (parsed := _parse_datetime(value)) is not None
    ]


def _domain(result: Any) -> str | None:
    raw = _first(result, ("domain", "url", "source_url", "canonical_url"))
    if raw is None:
        return None
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    host = parsed.hostname or parsed.netloc
    return host.casefold().removeprefix("www.") if host else None


def _risk(risk_type: str, severity: str, message: str, result_ids: list[str] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"type": risk_type, "severity": severity, "message": message}
    if result_ids is not None:
        row["result_ids"] = result_ids
    return row


def flag_context_risks(results: Iterable[Any], *, now: date | datetime | None = None) -> dict[str, Any]:
    """Return ordered risk flags for a retrieval context."""
    current = _normalize_now(now)
    rows = [
        {
            "result_id": _result_id(result, index),
            "domain": _domain(result),
            "source_project": _first(result, ("source_project",)),
            "content": _first(result, ("content", "text", "snippet")) or "",
            "dates": _dates(result),
        }
        for index, result in enumerate(results)
    ]

    risks: list[dict[str, Any]] = []
    if rows:
        latest_dates = [max(row["dates"]) for row in rows if row["dates"]]
        if latest_dates and all((current - item).days > 365 for item in latest_dates):
            risks.append(_risk("stale-evidence", "medium", "all dated evidence is older than one year"))

        known_domains = [row["domain"] for row in rows if row["domain"]]
        if len(rows) > 1 and len(set(known_domains)) <= 1:
            risks.append(_risk("single-source-dependence", "medium", "results rely on one source domain"))

        missing = [
            row["result_id"]
            for row in rows
            if not row["domain"] and not row["source_project"]
        ]
        if missing:
            risks.append(_risk("missing-provenance", "high", "some results lack source provenance", missing))

        conflicting = [
            row["result_id"]
            for row in rows
            if row["dates"] and (max(row["dates"]) - min(row["dates"])).days > 365
        ]
        if conflicting:
            risks.append(_risk("conflicting-dates", "medium", "some results contain dates more than a year apart", conflicting))

        short = [row["result_id"] for row in rows if len(row["content"]) < 80]
        if short:
            risks.append(_risk("short-content", "low", "some results have very short content", short))

    severity_order = {"high": 0, "medium": 1, "low": 2}
    risks.sort(key=lambda item: (severity_order[item["severity"]], item["type"]))
    return {
        "risk_count": len(risks),
        "risks": risks,
        "summary": {
            "result_count": len(rows),
            "source_domains": dict(sorted(Counter(row["domain"] or "unknown" for row in rows).items())),
        },
    }
