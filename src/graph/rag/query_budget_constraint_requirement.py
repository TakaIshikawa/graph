"""Detect budget constraint requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_TERM_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("budget", re.compile(r"\b(?:budget|budgeted|within\s+budget)\b", re.I)),
    ("price", re.compile(r"\b(?:price|pricing|priced|cost|costs|costing|expense|expenses|spend|spending)\b", re.I)),
    ("cost_ceiling", re.compile(r"\b(?:cost\s+ceiling|price\s+cap|cost\s+cap|max(?:imum)?\s+(?:cost|price|spend)|under\s+\$?\d|less\s+than\s+\$?\d|not\s+exceed)\b", re.I)),
    ("subscription_tier", re.compile(r"\b(?:subscription\s+tier|pricing\s+tier|plan\s+tier|starter\s+plan|pro\s+plan|enterprise\s+plan|basic\s+plan)\b", re.I)),
    ("free_or_paid", re.compile(r"\b(?:free\s+tier|free\s+plan|free\s+version|paid\s+plan|paid\s+tier|free\s+or\s+paid|no[-\s]?cost)\b", re.I)),
    ("token_budget", re.compile(r"\b(?:token\s+budget|token\s+limit|context\s+budget|context\s+window|input\s+tokens?|output\s+tokens?)\b", re.I)),
    ("time_budget", re.compile(r"\b(?:time\s+budget|within\s+\d+\s*(?:sec(?:ond)?s?|min(?:ute)?s?|hours?)|under\s+\d+\s*(?:sec(?:ond)?s?|min(?:ute)?s?|hours?)|latency\s+budget|response\s+time)\b", re.I)),
)

_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|JPY|CAD|AUD)\b)", re.I),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:k|m)?\s?(?:tokens?|tok)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|milliseconds?|sec(?:ond)?s?|min(?:ute)?s?|hours?|hrs?)\b", re.I),
    re.compile(r"\b(?:free\s+tier|free\s+plan|paid\s+tier|paid\s+plan|starter\s+plan|pro\s+plan|enterprise\s+plan|basic\s+plan)\b", re.I),
)


def detect_query_budget_constraint_requirement(query: str) -> dict[str, Any]:
    """Return budget constraint cues and extracted values for a query."""
    text = _normalize_query(query)
    constraint_terms = _constraint_terms(text)
    budget_values = _budget_values(text)
    requires = bool(constraint_terms or budget_values)
    return {
        "requires_budget_constraint": requires,
        "constraint_terms": constraint_terms,
        "budget_values": budget_values,
        "rationale": _rationale(constraint_terms, budget_values),
    }


def _constraint_terms(text: str) -> list[str]:
    return [term for term, pattern in _TERM_SPECS if pattern.search(text)]


def _budget_values(text: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for pattern in _VALUE_PATTERNS:
        for match in pattern.finditer(text):
            value = " ".join(match.group(0).split())
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            values.append(value)
    return sorted(values, key=lambda value: value.casefold())


def _rationale(constraint_terms: list[str], budget_values: list[str]) -> str:
    if not constraint_terms and not budget_values:
        return "No budget, pricing, token, tier, or time-budget constraints were detected."
    if constraint_terms and budget_values:
        return "Detected budget-related constraint language with explicit budget values."
    if budget_values:
        return "Detected explicit budget values that should constrain retrieval or answer planning."
    return "Detected budget-related constraint language without explicit numeric values."


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.strip().split())
