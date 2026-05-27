"""Audit evidence age against a staleness threshold."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import parse_date, result_id, rounded_ratio, string, value

_DATE_KEYS = ("published_at", "updated_at", "created_at", "date")


def audit_evidence_staleness(evidence: Iterable[Any], *, as_of: Any = None, stale_after_days: int = 365) -> dict[str, Any]:
    """Return stale, dated, and undated evidence counts."""
    if stale_after_days <= 0:
        raise ValueError("stale_after_days must be positive")
    reference = parse_date(as_of) if as_of is not None else date.today()
    if reference is None:
        raise ValueError("as_of must be a valid date")

    items = list(evidence or [])
    dated = stale = undated = 0
    findings = []
    for index, item in enumerate(items):
        item_id = result_id(item, index)
        raw = _date_value(item)
        parsed = parse_date(raw)
        if parsed is None:
            undated += 1
            finding_type = "invalid_date" if string(raw) else "undated_evidence"
            findings.append({"type": finding_type, "evidence_id": item_id, "value": string(raw)})
            continue
        dated += 1
        age_days = max(0, (reference - parsed).days)
        if age_days > stale_after_days:
            stale += 1
            findings.append({"type": "stale_evidence", "evidence_id": item_id, "date": parsed.isoformat(), "age_days": age_days})

    return {
        "evidence_count": len(items),
        "dated_evidence_count": dated,
        "stale_evidence_count": stale,
        "undated_evidence_count": undated,
        "stale_ratio": rounded_ratio(stale, len(items)),
        "findings": findings,
    }


def _date_value(item: Any) -> Any:
    for key in _DATE_KEYS:
        raw = value(item, key)
        if string(raw):
            return raw
    return None
