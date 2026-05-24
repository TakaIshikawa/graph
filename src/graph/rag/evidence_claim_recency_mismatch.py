"""Flag current claims supported only by stale evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import coerce_now, content_text, result_date, result_id

_CURRENT_RE = re.compile(r"\b(?:current(?:ly)?|now|today|latest|recent(?:ly)?|still|as\s+of)\b", re.I)
_STALE_DAYS = 365


def analyze_evidence_claim_recency_mismatch(answer: str, evidence: Iterable[Any], now: str | None = None) -> dict[str, Any]:
    """Return stale-evidence warnings for current or recent answer claims."""
    today = coerce_now(now)
    text = " ".join(str(answer or "").split())
    current_claim = bool(_CURRENT_RE.search(text))
    rows: list[dict[str, Any]] = []
    stale_support = False
    dated_count = 0
    for index, item in enumerate(evidence or []):
        parsed = result_date(item) or _date_from_text(content_text(item))
        age_days = (today - parsed).days if parsed else None
        stale = current_claim and age_days is not None and age_days > _STALE_DAYS
        stale_support = stale_support or stale
        dated_count += 1 if parsed else 0
        rows.append(
            {
                "result_id": result_id(item, index),
                "date": parsed.isoformat() if parsed else None,
                "age_days": age_days,
                "is_stale_for_current_claim": stale,
            }
        )
    warnings = []
    if current_claim and rows and dated_count == 0:
        warnings.append("current_claim_without_dated_evidence")
    if stale_support and not any(not row["is_stale_for_current_claim"] and row["date"] for row in rows):
        warnings.append("current_claim_supported_only_by_stale_evidence")
    return {"has_current_claim": current_claim, "now": today.isoformat(), "evidence": rows, "warnings": warnings}


def _date_from_text(text: str) -> date | None:
    match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(0))
    except ValueError:
        return None
