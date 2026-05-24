"""Audit tradeoff coverage in recommendation-style RAG answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_RECOMMEND_RE = re.compile(r"\b(?:recommend|should|best|choose|use|adopt|prioritize)\b", re.I)
_QUERY_RECOMMEND_RE = re.compile(r"\b(?:should|recommend|which|choose|best|option)\b", re.I)
_CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("benefits", re.compile(r"\b(?:benefits?|advantages?|improves?|gains?|upside|pros?)\b", re.I)),
    ("costs", re.compile(r"\b(?:costs?|prices?|expenses?|budget|fees?|spend)\b", re.I)),
    ("risks", re.compile(r"\b(?:risks?|downsides?|caveats?|failures?|concerns?)\b", re.I)),
    ("alternatives", re.compile(r"\b(?:alternatives?|instead|another\s+approach|fallback)\b", re.I)),
    ("effort", re.compile(r"\b(?:effort|implementation|migration|timeline|workload|complexity)\b", re.I)),
)


def audit_answer_tradeoff_coverage(answer: str, query: str = "", evidence: Iterable[Any] | None = None) -> dict[str, Any]:
    """Return tradeoff category coverage for recommendation answers."""
    text = " ".join(str(answer or "").split())
    is_recommendation = bool(_RECOMMEND_RE.search(text) or _QUERY_RECOMMEND_RE.search(str(query or "")))
    coverage = {name: bool(pattern.search(text)) for name, pattern in _CATEGORIES}
    missing = [name for name, covered in coverage.items() if is_recommendation and not covered]
    return {
        "is_recommendation_answer": is_recommendation,
        "coverage": coverage,
        "missing_tradeoff_categories": missing,
        "warnings": [f"missing_{name}" for name in missing],
    }
