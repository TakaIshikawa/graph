"""Audit decision criteria coverage in recommendation answers."""

from __future__ import annotations

import re
from typing import Any

_DECISION_RE = re.compile(r"\b(?:recommend|recommendation|choose|pick|prefer|best|which\s+option|compare|comparison)\b", re.I)
_CRITERIA_RE = re.compile(r"\b(?:choose\s+if|prefer\s+when|criteria|criterion|must\s+have|requirement|because|if\s+you\s+need)\b", re.I)
_THRESHOLD_RE = re.compile(r"(?:\b(?:threshold|at\s+least|under|over|minimum|maximum|no\s+more\s+than|greater\s+than|less\s+than)\b|\b\d+(?:\.\d+)?\s*%)", re.I)
_TRADEOFF_RE = re.compile(r"\b(?:trade[-\s]?off|however|but|versus|vs\.?|downside|upside|risk|cost|benefit)\b", re.I)
_TIE_BREAKER_RE = re.compile(r"\b(?:tie[-\s]?breaker|if\s+tied|when\s+both|all\s+else\s+equal|default\s+to|otherwise\s+choose)\b", re.I)


def audit_answer_decision_criteria(answer: str) -> dict[str, Any]:
    """Return criteria, threshold, tradeoff, and tie-breaker readiness signals."""
    text = _inline_text(answer)
    is_decision_answer = bool(_DECISION_RE.search(text))
    criteria_count = _count(_CRITERIA_RE, text)
    threshold_count = _count(_THRESHOLD_RE, text)
    tradeoff_count = _count(_TRADEOFF_RE, text)
    tie_breaker_count = _count(_TIE_BREAKER_RE, text)
    missing = []
    if is_decision_answer:
        if criteria_count == 0:
            missing.append("criteria")
        if threshold_count == 0:
            missing.append("thresholds")
        if tie_breaker_count == 0:
            missing.append("tie_breakers")
    score = 1.0 if not is_decision_answer else round((3 - len(missing)) / 3, 3)
    return {
        "criteria_count": criteria_count,
        "threshold_count": threshold_count,
        "tradeoff_count": tradeoff_count,
        "missing_decision_elements": missing,
        "decision_readiness_score": score,
    }


def _count(pattern: re.Pattern[str], text: str) -> int:
    return sum(1 for _ in pattern.finditer(text))


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
