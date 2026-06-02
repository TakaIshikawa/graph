"""Detect audit-log requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("admin_activity", "high", (r"\badmin(?:istrator)?\s+activity\s+logs?\b", r"\bprivileged\s+user\s+activity\s+logs?\b", r"\blog\s+admin\s+actions?\b")),
    ("audit_log", "high", (r"\baudit\s+logs?\b", r"\bauditability\b")),
    ("audit_trail", "high", (r"\baudit\s+trails?\b", r"\bevidence\s+trails?\b")),
    ("event_log", "medium", (r"\bevent\s+logs?\b", r"\bevent\s+history\b")),
    ("exportability", "medium", (r"\bexport(?:able|ability)\b", r"\bexport\s+(?:audit\s+)?logs?\b", r"\bdownload\s+(?:audit\s+)?logs?\b")),
    ("immutability", "high", (r"\bimmutable\s+(?:audit\s+)?logs?\b", r"\btamper[-\s]?proof\s+(?:audit\s+)?logs?\b", r"\btamper[-\s]?evident\s+(?:audit\s+)?logs?\b", r"\bworm\s+storage\b")),
    ("retention", "high", (r"\blog\s+retention\b", r"\baudit[-\s]?log\s+retention\b", r"\bretain\s+(?:audit\s+)?logs?\s+for\b")),
)


def detect_query_audit_log_requirement(query: str) -> dict[str, Any]:
    """Return audit-log requirements mentioned by a query."""
    text = _normalize_query(query)
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_audit_log_requirement": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
