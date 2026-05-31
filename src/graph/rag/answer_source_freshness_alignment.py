"""Align answer freshness claims with source date metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import parse_date, string, value

_FRESHNESS_RE = re.compile(r"(?i)\b(?:latest|current|currently|recent|recently|up to date|as of|outdated|stale)\b")
_CURRENT_RE = re.compile(r"(?i)\b(?:latest|current|currently|up to date|as of)\b")


def audit_answer_source_freshness_alignment(answer: str, sources: Iterable[Any]) -> dict[str, Any]:
    text = string(answer) or ""
    claims = [{"claim": match.group(0), "start": match.start()} for match in _FRESHNESS_RE.finditer(text)]
    dates = sorted(date_ for source in sources or [] if (date_ := _source_date(source)) is not None)
    warnings = []
    if claims and not dates:
        warnings.append("freshness_claim_without_source_dates")
    elif _CURRENT_RE.search(text) and dates and dates[-1].year < 2025:
        warnings.append("current_claim_with_stale_sources")

    score = 1.0
    if claims and not dates:
        score = 0.25
    elif warnings:
        score = 0.45
    elif claims and dates:
        score = 0.85

    return {
        "freshness_claims": claims,
        "newest_source_date": dates[-1].isoformat() if dates else None,
        "oldest_source_date": dates[0].isoformat() if dates else None,
        "stale_claim_warnings": warnings,
        "alignment_score": score,
    }


def _source_date(source: Any) -> date | None:
    for field in ("published_at", "updated_at", "date", "year"):
        parsed = _parse_date_or_year(value(source, field))
        if parsed:
            return parsed
    return None


def _parse_date_or_year(raw: Any) -> date | None:
    parsed = parse_date(raw)
    if parsed:
        return parsed
    text = string(raw)
    if text and re.fullmatch(r"\d{4}", text):
        return date(int(text), 1, 1)
    return None
