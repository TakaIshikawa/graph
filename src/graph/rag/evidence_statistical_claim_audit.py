"""Audit statistical claims in evidence snippets for missing context."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_STAT_RE = re.compile(
    r"(?<!\w)(?:"
    r"\d+(?:\.\d+)?\s?%"
    r"|\d+\s*(?:/|out\s+of)\s*\d+"
    r"|(?:average|mean|median)\s+\w+\s+(?:was|is|of)?\s*\d+(?:\.\d+)?\s+\w+"
    r"|rate\s+(?:was|is|of)?\s*\d+(?:\.\d+)?\s+\w+"
    r"|\d+(?:\.\d+)?\s+(?:cases|people|users|units|dollars?)"
    r")",
    re.I,
)
_TIME_RE = re.compile(r"\b(?:in|during|between|from|since|over|per)\s+(?:\d{4}|q[1-4]|january|february|march|april|may|june|july|august|september|october|november|december|year|month|week|day)\b", re.I)
_POP_RE = re.compile(r"\b(?:among|of|for|participants|patients|users|households|students|companies|sample)\b", re.I)
_UNIT_RE = re.compile(r"\b(?:%|percent|percentage|points|cases|people|users|units|kg|km|miles|hours|days|dollars?|usd|rate)\b", re.I)
_DENOM_RE = re.compile(r"\b(?:out\s+of|per|denominator|sample\s+of|n\s*=|/)\b", re.I)


def audit_evidence_statistical_claims(evidence_records: Iterable[Any]) -> dict[str, Any]:
    """Detect statistical claims and identify missing denominator, timeframe, population, or unit."""
    findings = []
    for index, record in enumerate(evidence_records):
        text = content_text(record)
        for match in _STAT_RE.finditer(text):
            window = text[max(0, match.start() - 90) : match.end() + 90]
            missing = []
            if not _DENOM_RE.search(window):
                missing.append("denominator")
            if not _TIME_RE.search(window):
                missing.append("timeframe")
            if not _POP_RE.search(window):
                missing.append("population")
            if not _UNIT_RE.search(window):
                missing.append("unit")
            findings.append(
                {
                    "evidence_id": result_id(record, index),
                    "claim_text": match.group(0),
                    "statistic_type": _stat_type(match.group(0), window),
                    "missing_context": missing,
                    "support_status": "sufficient" if not missing else "missing_context",
                }
            )
    return {"findings": findings}


def _stat_type(text: str, window: str = "") -> str:
    lowered = f"{text} {window}".casefold()
    if "%" in text or "percent" in lowered:
        return "percentage"
    if "/" in text or "out of" in lowered:
        return "ratio"
    if any(word in lowered for word in ("average", "mean", "median")):
        return "average"
    if "rate" in lowered:
        return "rate"
    return "count"
