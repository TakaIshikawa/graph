"""Detect RTO/RPO recovery objective requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(
    r"\b(?:rto|rpo|recovery\s+time\s+objective|recovery\s+point\s+objective|"
    r"maximum\s+tolerable\s+downtime|restore\s+time|acceptable\s+data\s+loss)\b",
    re.I,
)

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("data_loss_tolerance", "high", (r"\bacceptable\s+data\s+loss\b", r"\bdata\s+loss\s+tolerance\b")),
    ("downtime_tolerance", "high", (r"\bmaximum\s+tolerable\s+downtime\b", r"\bdowntime\s+tolerance\b")),
    ("restore_validation", "medium", (r"\brestore\s+validation\b", r"\brestore\s+tests?\b", r"\brecovery\s+tests?\b")),
    ("rpo", "high", (r"\brpo\b", r"\brecovery\s+point\s+objective\b")),
    ("rto", "high", (r"\brto\b", r"\brecovery\s+time\s+objective\b", r"\brestore\s+time\b")),
)


def detect_query_rto_rpo_requirements(query: str) -> dict[str, Any]:
    """Return detected recovery objective requirements for a query."""
    normalized = _normalize_query(query)
    if not _CONTEXT_RE.search(normalized):
        return {"has_rto_rpo_requirements": False, "requirements": [], "normalized_query": normalized}

    requirements = []
    for category, severity, patterns in _CATEGORY_SPECS:
        match = _first_match(normalized, patterns)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})

    requirements.sort(key=lambda row: row["category"])
    return {
        "has_rto_rpo_requirements": bool(requirements),
        "requirements": requirements,
        "normalized_query": normalized,
    }


def _first_match(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
