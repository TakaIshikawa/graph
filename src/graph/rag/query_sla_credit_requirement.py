"""Detect SLA credit requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CREDIT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("service_credit", (r"\bservice\s+credits?\b",)),
    ("sla_penalty", (r"\bsla\s+penalt(?:y|ies)\b", r"\bpenalty\s+for\s+sla\b")),
    ("uptime_credit", (r"\buptime\s+credits?\b",)),
    ("refund_for_downtime", (r"\brefund\s+for\s+downtime\b", r"\bdowntime\s+refund\b")),
    ("credit_schedule", (r"\bcredit\s+schedule\b",)),
    ("breach_compensation", (r"\bbreach\s+of\s+sla\b", r"\bsla\s+breach\s+compensation\b")),
)
_REMEDY_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("refund", (r"\brefund\b",)),
    ("penalty", (r"\bpenalt(?:y|ies)\b",)),
    ("remedy", (r"\bremed(?:y|ies)\b",)),
    ("compensation", (r"\bcompensation\b",)),
)


def detect_query_sla_credit_requirement(query: str) -> dict[str, Any]:
    """Return SLA credit terms mentioned by a query."""
    text = " ".join(str(query or "").split())
    credit_terms = [term for term, patterns in _CREDIT_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    remedy_terms = [term for term, patterns in _REMEDY_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    recommendations = ["review SLA credit clause"] if credit_terms else []
    return {
        "requires_sla_credit_terms": bool(credit_terms),
        "credit_terms": credit_terms,
        "matched_phrases": credit_terms + remedy_terms,
        "remedy_terms": remedy_terms,
        "recommendations": recommendations,
        "confidence": "high" if credit_terms else "none",
    }
