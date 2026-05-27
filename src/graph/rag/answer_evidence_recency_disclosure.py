"""Audit whether answers disclose stale or mixed-age evidence."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from graph.rag._analysis_utils import coerce_now, result_date

_DISCLOSURE_RE = re.compile(
    r"\b(?:as of|through|between \d{4}|from \d{4}|older evidence|outdated|stale|mixed[- ]age|date range|recent and older)\b",
    re.I,
)


def audit_answer_evidence_recency_disclosure(
    answer: str,
    evidence_items: list[dict],
    *,
    now: Any = None,
    stale_after_days: int = 1095,
    mixed_span_days: int = 730,
) -> dict[str, Any]:
    today = coerce_now(now)
    dates = [day for item in evidence_items or [] if (day := result_date(item)) is not None]
    oldest = min(dates) if dates else None
    newest = max(dates) if dates else None
    has_stale = any((today - day).days > stale_after_days for day in dates)
    has_mixed_age = bool(oldest and newest and (newest - oldest).days > mixed_span_days)
    disclosure_present = bool(_DISCLOSURE_RE.search(str(answer or "")))
    needs_disclosure = has_stale or has_mixed_age
    return {
        "needs_recency_disclosure": needs_disclosure,
        "has_recency_disclosure": disclosure_present,
        "passes": not needs_disclosure or disclosure_present,
        "date_count": len(dates),
        "missing_date_count": len(evidence_items or []) - len(dates),
        "oldest_date": _iso(oldest),
        "newest_date": _iso(newest),
        "has_stale_evidence": has_stale,
        "has_mixed_age_evidence": has_mixed_age,
    }


def _iso(day: date | None) -> str | None:
    return day.isoformat() if day else None
