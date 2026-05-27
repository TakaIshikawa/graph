"""Detect cost and retrieval budget requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_LOW_RE = re.compile(r"\b(?:cheap|free|low[- ]cost|minimi[sz]e cost|save tokens?|minimal tokens?|quick pass)\b", re.I)
_HIGH_RE = re.compile(r"\b(?:exhaustive|comprehensive|high[- ]quality|deep research|do not miss|maximum coverage)\b", re.I)
_LIMIT_RE = re.compile(r"\b(?:under|below|less than|no more than|up to|limit(?:ed)? to|max(?:imum)? of)?\s*(\d[\d,]*)\s*(tokens?|sources?|documents?|docs?)\b", re.I)


def detect_query_cost_requirements(query: str) -> dict[str, Any]:
    text = " ".join(("" if query is None else str(query)).split())
    low = _matches(_LOW_RE, text)
    high = _matches(_HIGH_RE, text)
    limits = [
        {"amount": int(m.group(1).replace(",", "")), "unit": _unit(m.group(2)), "text": m.group(0).strip()}
        for m in _LIMIT_RE.finditer(text)
    ]
    if low and not high:
        level = "low"
        guidance = "Prefer concise retrieval, free/open sources, and respect explicit limits."
    elif high:
        level = "high_coverage"
        guidance = "Prioritize coverage and quality; use limits only as hard caps."
    elif limits:
        level = "limited"
        guidance = "Respect explicit retrieval limits."
    else:
        level = "unspecified"
        guidance = "Use default retrieval depth."
    cues = [{"family": "low_cost", "text": m["text"]} for m in low] + [{"family": "high_coverage", "text": m["text"]} for m in high]
    return {
        "cost_sensitive": bool(low or limits),
        "budget_level": level,
        "matched_cues": cues,
        "requested_limits": limits,
        "confidence": 0.8 if cues or limits else 0.0,
        "retrieval_guidance": guidance,
    }


def _matches(pattern: re.Pattern[str], text: str) -> list[dict[str, Any]]:
    return [{"text": m.group(0), "start": m.start(), "end": m.end()} for m in pattern.finditer(text)]


def _unit(unit: str) -> str:
    normalized = unit.casefold()
    if normalized.startswith("token"):
        return "tokens"
    if normalized.startswith("source"):
        return "sources"
    return "documents"
