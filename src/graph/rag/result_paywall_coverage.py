"""Analyze result access metadata for paywall coverage."""

from __future__ import annotations

from collections import Counter
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, rounded_ratio, string, value

_OPEN = ("open", "open access", "free", "public", "full text")
_PAYWALLED = ("paywall", "paid", "subscription", "purchase")
_LOGIN = ("login", "log in", "sign in", "account required", "registration")


def analyze_result_paywall_coverage(results: list[dict]) -> dict[str, Any]:
    rows = list(results or [])
    counts: Counter[str] = Counter({"open": 0, "paywalled": 0, "login_required": 0, "unknown": 0})
    paywalled_ids: list[str] = []
    classified = []
    for index, result in enumerate(rows):
        status = _status(result)
        rid = result_id(result, index)
        counts[status] += 1
        if status == "paywalled":
            paywalled_ids.append(rid)
        classified.append({"id": rid, "access_status": status})
    total = len(rows)
    return {
        "total_results": total,
        "counts": dict(counts),
        "ratios": {key: rounded_ratio(counts[key], total) for key in ("open", "paywalled", "login_required", "unknown")},
        "paywalled_result_ids": paywalled_ids,
        "results": classified,
    }


def _status(result: Any) -> str:
    if value(result, "is_open_access") is True or value(result, "open_access") is True:
        return "open"
    if value(result, "paywall") is True or value(result, "is_paywalled") is True:
        return "paywalled"
    text = " ".join(
        part
        for part in [
            string(value(result, "access")) or "",
            string(value(result, "access_status")) or "",
            string(value(result, "notes")) or "",
            string(value(result, "url")) or "",
            content_text(result),
        ]
        if part
    ).casefold()
    if any(cue in text for cue in _LOGIN):
        return "login_required"
    if any(cue in text for cue in _PAYWALLED):
        return "paywalled"
    if any(cue in text for cue in _OPEN):
        return "open"
    return "unknown"
