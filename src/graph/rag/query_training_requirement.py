"""Detect training and enablement requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("documentation", "medium", (r"\bdocumentation\b", r"\bdocs\b", r"\buser\s+guide\b")),
    ("onboarding", "medium", (r"\bonboarding\b", r"\bnew\s+hire\s+enablement\b")),
    ("playbook", "medium", (r"\bplaybooks?\b",)),
    ("runbook", "high", (r"\brunbooks?\b",)),
    ("training_plan", "high", (r"\btraining\s+plans?\b", r"\btraining\s+programs?\b")),
    ("user_education", "medium", (r"\buser\s+education\b", r"\bteach\s+users?\b")),
)


def detect_query_training_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
