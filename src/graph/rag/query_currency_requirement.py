"""Detect currency normalization requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CURRENCY_CODES = ("USD", "EUR", "GBP", "JPY", "CAD", "AUD")
_SYMBOLS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("USD", re.compile(r"\$")),
    ("EUR", re.compile(r"€")),
    ("GBP", re.compile(r"£")),
    ("JPY", re.compile(r"¥")),
)
_CONVERSION: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("exchange_rate", re.compile(r"\bexchange rate\b", re.I)),
    ("local_currency", re.compile(r"\blocal currency\b", re.I)),
    ("convert_currency", re.compile(r"\bconvert(?:ed)?\b", re.I)),
    ("purchasing_power", re.compile(r"\bpurchasing power\b", re.I)),
)
_ADJUSTMENT: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("inflation_adjusted", re.compile(r"\binflation[- ]adjusted\b", re.I)),
    ("nominal", re.compile(r"\bnominal\b", re.I)),
    ("real_terms", re.compile(r"\breal terms\b|\breal dollars\b", re.I)),
)


def detect_query_currency_requirement(query: str) -> dict[str, Any]:
    """Return currency cues and normalization recommendations."""
    normalized = _normalize_query(query)
    explicit = _explicit_currencies(query)
    conversion = [label for label, pattern in _CONVERSION if pattern.search(normalized)]
    adjustment = [label for label, pattern in _ADJUSTMENT if pattern.search(normalized)]
    requires = bool(explicit or conversion or adjustment)
    recommendations = []
    if explicit or conversion:
        recommendations.append("normalize_monetary_values_to_requested_or_reference_currency")
    if adjustment:
        recommendations.append("separate_nominal_values_from_inflation_adjusted_real_terms")
    return {
        "requires_currency_normalization": requires,
        "explicit_currencies": explicit,
        "conversion_cues": conversion,
        "adjustment_cues": adjustment,
        "recommendations": recommendations,
        "confidence": 0.9 if explicit else (0.75 if conversion or adjustment else 0.0),
        "normalized_query": normalized,
    }


def _explicit_currencies(query: str) -> list[str]:
    found: set[str] = set()
    for code in _CURRENCY_CODES:
        if re.search(rf"\b{code}\b", query, re.I):
            found.add(code)
    for code, pattern in _SYMBOLS:
        if pattern.search(query):
            found.add(code)
    return [code for code in _CURRENCY_CODES if code in found]


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.casefold().split())
