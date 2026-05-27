"""Detect metric requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_METRIC_PATTERNS = {
    "kpi": r"\bkpis?\b|\bkey performance indicators?\b",
    "benchmark": r"\bbenchmarks?\b",
    "threshold": r"\bthresholds?\b|\bat least\b|\bminimum\b|\bmaximum\b",
    "rate": r"\brates?\b",
    "percentage": r"\bpercent(?:age|ages)?\b|%",
    "latency": r"\blatency\b|\bresponse time\b",
    "cost": r"\bcosts?\b|\bprice\b|\bspend\b",
    "accuracy": r"\baccuracy\b|\baccurate\b",
    "recall": r"\brecall\b",
    "roi": r"\broi\b|\breturn on investment\b",
    "metric": r"\bmetrics?\b|\bmeasur(?:e|able|ement)s?\b",
}
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|ms|sec(?:onds?)?|s|usd|dollars?|x)?", re.I)


def detect_query_metric_requirement(query: str) -> dict[str, Any]:
    text = str(query or "")
    metric_terms = [_normalize(term) for term, pattern in _METRIC_PATTERNS.items() if re.search(pattern, text, re.I)]
    numeric_cues = _dedupe(match.group(0).strip() for match in _NUMBER_RE.finditer(text))
    required = bool(metric_terms or numeric_cues)
    return {
        "required": required,
        "metric_terms": metric_terms,
        "numeric_cues": numeric_cues if required else [],
        "suggested_evidence_fields": _fields(metric_terms) if required else [],
    }


def _fields(terms: list[str]) -> list[str]:
    fields = ["metric_name", "metric_value", "unit"]
    if "threshold" in terms:
        fields.append("threshold")
    if "benchmark" in terms or "kpi" in terms:
        fields.append("benchmark")
    return _dedupe(fields)


def _normalize(value: str) -> str:
    return value.casefold().replace(" ", "_")


def _dedupe(values: Any) -> list[str]:
    seen = set()
    out = []
    for value in values:
        key = str(value).casefold()
        if key and key not in seen:
            seen.add(key)
            out.append(str(value))
    return out
