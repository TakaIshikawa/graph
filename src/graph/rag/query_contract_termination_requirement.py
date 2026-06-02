"""Detect contract termination requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(r"\b(?:contract|vendor|procurement|msa|agreement|subscription|supplier)\b", re.I)
_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("data_return_deletion", "high", (r"\bpost[-\s]?termination\s+data\s+(?:return|deletion)\b", r"\bdata\s+return\s+or\s+deletion\b")),
    ("exit_fees", "medium", (r"\bexit\s+fees?\b", r"\btermination\s+fees?\b")),
    ("notice_period", "high", (r"\btermination\s+notice\s+period\b", r"\b\d+\s+days?\s+notice\b")),
    ("survival_clauses", "medium", (r"\bsurvival\s+clauses?\b", r"\bclauses?\s+survive\s+termination\b")),
    ("termination_for_cause", "high", (r"\btermination\s+for\s+cause\b", r"\bterminate\s+for\s+cause\b")),
    ("termination_for_convenience", "high", (r"\btermination\s+for\s+convenience\b", r"\bterminate\s+for\s+convenience\b")),
    ("termination_rights", "high", (r"\btermination\s+rights?\b", r"\bright\s+to\s+terminate\b")),
    ("transition_assistance", "medium", (r"\btransition\s+assistance\b", r"\btermination\s+transition\s+support\b")),
)


def detect_query_contract_termination_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if not _CONTEXT_RE.search(text):
        return []
    rows = []
    for category, requirement_strength, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append(
                {
                    "matched_text": match.group(0),
                    "category": category,
                    "requirement_strength": requirement_strength,
                }
            )
    return sorted(rows, key=lambda row: row["category"])
