"""Analyze traceability signals for claims in retrieved results."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_date, result_id, string, value

_CITATION_RE = re.compile(r"(?:\[[0-9]+\]|\([A-Z][A-Za-z]+,\s*(?:19|20)\d{2}\)|doi:|https?://)", re.I)
_QUOTE_RE = re.compile(r'"[^"\n]{8,}"|“[^”\n]{8,}”')
_SIGNALS = ("citation", "url", "quote", "title", "author", "date")


def analyze_result_claim_traceability(results: Iterable[Any]) -> dict[str, Any]:
    items = list(results or [])
    result_reports = [_analyze(item, index) for index, item in enumerate(items)]
    return {
        "total_results": len(items),
        "weak_traceability_count": sum(1 for report in result_reports if report["traceability_score"] < 3),
        "results": result_reports,
    }


def _analyze(result: Any, index: int) -> dict[str, Any]:
    text = content_text(result) or (string(result) if isinstance(result, str) else "") or ""
    present = {
        "citation": bool(_CITATION_RE.search(text)),
        "url": bool(_first_text(result, ("url", "source_url", "canonical_url", "link"))) or bool(re.search(r"https?://", text)),
        "quote": bool(_QUOTE_RE.search(text)),
        "title": bool(_first_text(result, ("title", "headline", "name"))),
        "author": bool(_first_text(result, ("author", "creator", "byline"))),
        "date": result_date(result) is not None or bool(_first_text(result, ("date", "published_at", "updated_at"))),
    }
    signals = [signal for signal in _SIGNALS if present[signal]]
    missing = [signal for signal in _SIGNALS if not present[signal]]
    return {
        "result_id": result_id(result, index),
        "traceability_score": len(signals),
        "traceability_signals": signals,
        "missing_signals": missing,
    }


def _first_text(result: Any, keys: tuple[str, ...]) -> str:
    for key in keys:
        text = string(value(result, key))
        if text:
            return text
    return ""
