"""Audit cited answer sources for freshness issues."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any

_DATE_KEYS = ("source_date", "published_at", "updated_at", "date")
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def audit_answer_citation_freshness(
    answer: str,
    citations: Iterable[Any],
    *,
    reference_date: date | datetime | str | None = None,
    stale_after_days: int = 365,
    aging_after_days: int = 180,
) -> list[dict[str, Any]]:
    reference = _parse_date(reference_date) or datetime.now(timezone.utc).date()
    cited_tokens = set(_citation_tokens(answer))
    rows: list[dict[str, Any]] = []
    for index, citation in enumerate(citations):
        citation_id = _citation_id(citation, index)
        if cited_tokens and citation_id not in cited_tokens:
            continue
        raw_date = _first(citation, _DATE_KEYS)
        source_date = _parse_date(raw_date)
        if source_date is None:
            rows.append({"citation_id": citation_id, "source_date": "", "age_days": None, "severity": "high", "reason": "missing_or_invalid_source_date"})
            continue
        age_days = max(0, (reference - source_date).days)
        if age_days > stale_after_days:
            severity, reason = "high", "citation_source_is_stale"
        elif age_days > aging_after_days:
            severity, reason = "medium", "citation_source_is_aging"
        else:
            severity, reason = "low", "citation_source_is_fresh"
        rows.append({"citation_id": citation_id, "source_date": source_date.isoformat(), "age_days": age_days, "severity": severity, "reason": reason})
    return sorted(rows, key=lambda row: (_SEVERITY_RANK[row["severity"]], row["citation_id"]))


def _citation_tokens(answer: str) -> list[str]:
    import re

    return re.findall(r"\[(?:cite:)?([A-Za-z0-9_.:-]+)\]", str(answer))


def _citation_id(citation: Any, index: int) -> str:
    return str(_first(citation, ("citation_id", "id", "source_id")) or f"citation-{index + 1}")


def _first(item: Any, keys: tuple[str, ...]) -> Any:
    for container in (item, _value(item, "metadata")):
        if container is None:
            continue
        for key in keys:
            value = _value(container, key)
            if value not in (None, ""):
                return value
    return None


def _value(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            return None
