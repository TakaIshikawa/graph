"""Audit retrieved results for conflicting date signals."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import parse_date, string, value

_DATE_FIELDS = ("published_at", "published", "publication_date", "updated_at", "updated", "accessed_at", "accessed", "retrieved_at", "date", "year")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")


def audit_result_conflicting_dates(results: Iterable[Any]) -> dict[str, Any]:
    rows = list(results or [])
    conflicts: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    affected: set[int] = set()

    for index, result in enumerate(rows):
        dates = _date_signals(result)
        for field in dates:
            counts[field] += len(dates[field])
        published = _first(dates, ("published_at", "published", "publication_date"))
        updated = _first(dates, ("updated_at", "updated"))
        if published and updated and updated < published:
            conflicts.append(_conflict(index, "updated_before_published", published.isoformat(), updated.isoformat()))
            affected.add(index)

        metadata_years = {item.year for field, values in dates.items() if field != "content_year" for item in values}
        content_years = {item.year for item in dates.get("content_year", [])}
        years = sorted(metadata_years | content_years)
        if content_years and metadata_years and len(years) > 1:
            conflicts.append(_conflict(index, "conflicting_years", str(years[0]), str(years[-1])))
            affected.add(index)

    return {
        "conflicts": conflicts,
        "date_field_counts": dict(sorted(counts.items())),
        "affected_result_count": len(affected),
        "examples": conflicts[:5],
    }


def _date_signals(result: Any) -> dict[str, list[date]]:
    dates: dict[str, list[date]] = {}
    for field in _DATE_FIELDS:
        parsed = _parse_date_or_year(value(result, field))
        if parsed:
            dates[field] = [parsed]
    text = " ".join(filter(None, [string(value(result, "snippet")), string(value(result, "summary")), string(value(result, "content")), string(value(result, "text"))]))
    content_dates = [date(int(match.group(1)), 1, 1) for match in _YEAR_RE.finditer(text)]
    if content_dates:
        dates["content_year"] = content_dates
    return dates


def _parse_date_or_year(raw: Any) -> date | None:
    parsed = parse_date(raw)
    if parsed:
        return parsed
    text = string(raw)
    if text and re.fullmatch(r"\d{4}", text):
        return date(int(text), 1, 1)
    return None


def _first(dates: dict[str, list[date]], fields: tuple[str, ...]) -> date | None:
    for field in fields:
        if dates.get(field):
            return dates[field][0]
    return None


def _conflict(index: int, conflict_type: str, first: str, second: str) -> dict[str, Any]:
    return {"result_index": index, "conflict_type": conflict_type, "first_value": first, "second_value": second}
