"""Extract normalized temporal anchors from queries."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime
from typing import Any

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_RANGE_RE = re.compile(r"((?:19|20)\d{2})\s*(?:-|to|through)\s*((?:19|20)\d{2})", re.IGNORECASE)
_QUARTER_RE = re.compile(r"\bQ([1-4])\s*((?:19|20)\d{2})\b|\b((?:19|20)\d{2})\s*Q([1-4])\b", re.IGNORECASE)
_MONTH_RE = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+((?:19|20)\d{2})\b", re.IGNORECASE)


def detect_query_temporal_anchors(query: str, *, reference_date: date | datetime | str | None = None) -> list[dict[str, Any]]:
    text = str(query)
    reference = _parse_reference(reference_date) or date.today()
    anchors: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for match in _RANGE_RE.finditer(text):
        start, end = match.group(1), match.group(2)
        consumed.update((start, end))
        anchors.append({"type": "year_range", "text": match.group(0), "start": f"{start}-01-01", "end": f"{end}-12-31"})
    for match in _QUARTER_RE.finditer(text):
        quarter = int(match.group(1) or match.group(4))
        year = int(match.group(2) or match.group(3))
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        anchors.append({"type": "quarter", "text": match.group(0), "start": f"{year}-{start_month:02d}-01", "end": f"{year}-{end_month:02d}-{calendar.monthrange(year, end_month)[1]:02d}"})
    for match in _MONTH_RE.finditer(text):
        month = list(calendar.month_name).index(match.group(1).title())
        year = int(match.group(2))
        anchors.append({"type": "month", "text": match.group(0), "start": f"{year}-{month:02d}-01", "end": f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"})
    for match in _YEAR_RE.finditer(text):
        year = match.group(1)
        if year not in consumed and not any(anchor["text"].find(year) >= 0 and anchor["type"] in {"quarter", "month"} for anchor in anchors):
            anchors.append({"type": "year", "text": year, "start": f"{year}-01-01", "end": f"{year}-12-31"})
    anchors.extend(_relative(text, reference))
    return sorted(anchors, key=lambda row: (row["start"], row["type"], row["text"]))


def _relative(text: str, reference: date) -> list[dict[str, Any]]:
    lower = text.casefold()
    rows = []
    if "last year" in lower:
        year = reference.year - 1
        rows.append({"type": "relative", "text": "last year", "start": f"{year}-01-01", "end": f"{year}-12-31"})
    if "next quarter" in lower:
        quarter = ((reference.month - 1) // 3) + 2
        year = reference.year + (1 if quarter > 4 else 0)
        quarter = 1 if quarter > 4 else quarter
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        rows.append({"type": "relative", "text": "next quarter", "start": f"{year}-{start_month:02d}-01", "end": f"{year}-{end_month:02d}-{calendar.monthrange(year, end_month)[1]:02d}"})
    return rows


def _parse_reference(value: date | datetime | str | None) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value in (None, ""):
        return None
    return date.fromisoformat(str(value))
