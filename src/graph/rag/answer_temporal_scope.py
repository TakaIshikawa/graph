"""Audit temporal scope claims in draft RAG answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import result_date, string

_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{4})\b")


def audit_answer_temporal_scope(answer: str, query: str, results: Iterable[Any]) -> dict[str, Any]:
    """Return temporal claims, evidence bounds, warnings, and a conservative status."""
    answer_text = string(answer) or ""
    query_text = string(query) or ""
    claims = [_claim(match.group(0)) for match in _DATE_RE.finditer(answer_text)]
    dates = sorted(date_ for result in list(results or []) if (date_ := result_date(result)) is not None)
    bounds = {
        "oldest_date": dates[0].isoformat() if dates else None,
        "newest_date": dates[-1].isoformat() if dates else None,
        "dated_result_count": len(dates),
    }
    expectations = _expectations(query_text)
    warnings = []
    if not answer_text:
        warnings.append("empty_answer")
    if not dates:
        warnings.append("no_dated_evidence")
    for claim in claims:
        claim_date = claim["date"]
        if dates and claim_date < dates[0]:
            warnings.append("answer_date_before_evidence")
        if dates and claim_date > dates[-1]:
            warnings.append("answer_date_after_evidence")
        if "after_year" in expectations and claim_date.year < expectations["after_year"]:
            warnings.append("answer_date_before_query_scope")
        if "before_year" in expectations and claim_date.year > expectations["before_year"]:
            warnings.append("answer_date_after_query_scope")
    if expectations.get("requires_current") and dates and (date.today() - dates[-1]).days > 370:
        warnings.append("latest_query_with_old_evidence")
    warnings = _unique(warnings)
    return {
        "date_claims": claims,
        "evidence_date_bounds": bounds,
        "query_expectations": expectations,
        "warnings": warnings,
        "status": "warning" if warnings else "ok",
    }


def _claim(text: str) -> dict[str, Any]:
    if len(text) == 4:
        return {"text": text, "kind": "year", "date": date(int(text), 1, 1)}
    return {"text": text, "kind": "date", "date": date.fromisoformat(text)}


def _expectations(query: str) -> dict[str, Any]:
    lowered = query.casefold()
    data: dict[str, Any] = {}
    if any(cue in lowered for cue in ("latest", "current", "recent", "today", "now")):
        data["requires_current"] = True
    if match := re.search(r"\bafter\s+(\d{4})\b", lowered):
        data["after_year"] = int(match.group(1))
    if match := re.search(r"\bbefore\s+(\d{4})\b", lowered):
        data["before_year"] = int(match.group(1))
    return data


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
