"""Detect comparison axes in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_AXES = {
    "cost": ("cost", "price", "pricing", "cheap", "expensive", "budget"),
    "latency": ("latency", "speed", "fast", "slow", "response time"),
    "accuracy": ("accuracy", "quality", "precision", "recall", "correct"),
    "safety": ("safety", "risk", "secure", "harm", "guardrail"),
    "privacy": ("privacy", "private", "pii", "data retention", "tracking"),
    "maintainability": ("maintainability", "maintenance", "operability", "support", "complexity"),
    "usability": ("usability", "ux", "ease of use", "user friendly", "onboarding"),
}
_COMPARISON_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|better|best|trade[- ]?off|choose|which)\b", re.IGNORECASE)


def detect_query_comparison_axes(query: str) -> dict[str, Any]:
    text = str(query)
    axes = []
    for axis, terms in _AXES.items():
        matched = sorted({term for term in terms if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE)})
        if matched:
            axes.append({"axis": axis, "matched_terms": matched, "confidence": 0.9 if _COMPARISON_RE.search(text) else 0.65})
    return {"axes": axes, "confidence": max((axis["confidence"] for axis in axes), default=0.1 if text.strip() else 0.0)}
