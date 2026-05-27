"""Detect comparison baseline requirements in queries."""

from __future__ import annotations

import re
from typing import Any

_PHRASES = ("compared with", "compared to", "versus", "vs", "relative to", "before and after", "against baseline", "a/b")


def detect_query_comparison_baselines(query: str) -> dict[str, Any]:
    text = str(query or "")
    lowered = text.casefold()
    matched = [phrase for phrase in _PHRASES if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered)]
    baselines = []
    for phrase in matched:
        baselines.extend(_adjacent_terms(text, phrase))
    return {
        "requires_comparison": bool(matched),
        "comparison_terms": _dedupe(_normalize(p) for p in matched),
        "baseline_terms": _dedupe(baselines),
        "matched_phrases": matched,
    }


def _adjacent_terms(text: str, phrase: str) -> list[str]:
    pattern = re.compile(rf"(.{{0,40}})\b{re.escape(phrase)}\b(.{{0,40}})", re.I)
    match = pattern.search(text)
    if not match:
        return []
    terms = []
    for side in match.groups():
        words = re.findall(r"\b[a-z][a-z0-9-]{2,}\b", side.casefold())
        terms.extend(word for word in words[-3:] if word not in {"compare", "compared", "with", "and", "before", "after"})
    return terms


def _normalize(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _dedupe(values: Any) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
