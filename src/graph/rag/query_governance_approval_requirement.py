"""Detect governance approval requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("approval", "medium", (r"\bapprovals?\b", r"\bapproved\s+by\b", r"\bsign[-\s]?off\b")),
    ("change_review", "high", (r"\bchange\s+advisory\s+board\b", r"\bcab\s+review\b", r"\bchange\s+review\b")),
    ("legal_review", "high", (r"\blegal\s+review\b", r"\bcounsel\s+review\b")),
    ("procurement", "medium", (r"\bprocurement\b", r"\bvendor\s+approval\b", r"\bpurchasing\s+approval\b")),
    ("security_review", "high", (r"\bsecurity\s+review\b", r"\bsecurity\s+approval\b")),
    ("stakeholder_signoff", "medium", (r"\bstakeholder\s+sign[-\s]?off\b", r"\bstakeholder\s+approval\b")),
)


def detect_query_governance_approval_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if not text:
        return []

    rows: list[dict[str, Any]] = []
    for category, severity, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
