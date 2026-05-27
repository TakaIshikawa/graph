"""Detect cost sensitivity requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_BUDGET_RE = re.compile(
    r"\b(?:under|below|less\s+than|no\s+more\s+than|up\s+to|within|budget(?:ed)?\s+at)\s+"
    r"((?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?)|(?:\d[\d,]*(?:\.\d+)?\s*(?:USD|EUR|GBP|JPY|CAD|AUD)))\b",
    re.I,
)
_CURRENCY_RE = re.compile(r"(?:[$€£¥]|\b(?:USD|EUR|GBP|JPY|CAD|AUD)\b)", re.I)
_CUE_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("free", "free", re.compile(r"\b(?:free|no[-\s]?cost|open[-\s]?source|gratis)\b", re.I)),
    ("low_cost", "low_cost", re.compile(r"\b(?:cheap|low[-\s]?cost|affordable|inexpensive)\b", re.I)),
    ("budget_capped", "budget_capped", re.compile(r"\b(?:under|below|less\s+than|no\s+more\s+than|up\s+to|within\s+budget|budget\s+cap)\b", re.I)),
    ("total_cost_aware", "total_cost_aware", re.compile(r"\b(?:total\s+cost|tco|all[-\s]?in|lifetime\s+cost|enterprise\s+pricing|pricing\s+tier)\b", re.I)),
)
_MODE_PRIORITY = ("budget_capped", "free", "low_cost", "total_cost_aware")


def detect_query_cost_sensitivity_requirement(query: str) -> dict[str, Any]:
    """Return cost awareness cues, modes, amounts, and currency mentions."""
    text = _inline_text(query)
    cues = _cues(text)
    budget_amounts = _budget_amounts(text)
    currencies = _currency_mentions(text)
    modes = {cue["cost_mode"] for cue in cues}
    if budget_amounts:
        modes.add("budget_capped")
    cost_mode = next((mode for mode in _MODE_PRIORITY if mode in modes), "none")
    return {
        "requires_cost_awareness": bool(cues or budget_amounts or currencies),
        "cost_mode": cost_mode,
        "budget_amounts": budget_amounts,
        "currency_mentions": currencies,
        "cues": cues,
    }


def _cues(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue_type, mode, pattern in _CUE_SPECS:
        for match in pattern.finditer(text):
            rows.append({"cue": match.group(0), "type": cue_type, "cost_mode": mode, "span": [match.start(), match.end()]})
    return sorted(rows, key=lambda row: (row["span"][0], row["span"][1], row["type"]))


def _budget_amounts(text: str) -> list[dict[str, Any]]:
    return [
        {"amount": match.group(1).strip(), "cue": match.group(0).strip(), "span": [match.start(), match.end()]}
        for match in _BUDGET_RE.finditer(text)
    ]


def _currency_mentions(text: str) -> list[str]:
    seen: set[str] = set()
    mentions: list[str] = []
    for match in _CURRENCY_RE.finditer(text):
        value = match.group(0).upper()
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            mentions.append(value)
    return mentions


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
