"""Detect threat-modeling requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"\bthreat\s+model(?:ing|ling)?\b",
    r"\bstride\b",
    r"\battack\s+trees?\b",
    r"\bmisuse\s+cases?\b",
    r"\babuse\s+cases?\b",
    r"\btrust\s+boundar(?:y|ies)\b",
    r"\bdata\s+flow\s+diagrams?\b",
    r"\bdfds?\b",
    r"\bmitigation\s+mapping\b",
)

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "methodology",
        "high",
        (
            r"\bthreat\s+model(?:ing|ling)?\b",
            r"\bstride\b",
            r"\battack\s+trees?\b",
            r"\bmisuse\s+cases?\b",
            r"\babuse\s+cases?\b",
        ),
    ),
    (
        "assets",
        "medium",
        (
            r"\bassets?\b",
            r"\bsecurity\s+critical\s+assets?\b",
            r"\bdata\s+stores?\b",
            r"\bsensitive\s+data\s+flows?\b",
        ),
    ),
    (
        "trust_boundaries",
        "high",
        (
            r"\btrust\s+boundar(?:y|ies)\b",
            r"\bprivilege\s+boundar(?:y|ies)\b",
            r"\bsecurity\s+boundar(?:y|ies)\b",
            r"\bdata\s+flow\s+diagrams?\b",
            r"\bdfds?\b",
        ),
    ),
    (
        "threats",
        "high",
        (
            r"\bthreats\b",
            r"\battack\s+paths?\b",
            r"\battack\s+vectors?\b",
            r"\babuse\s+scenarios?\b",
        ),
    ),
    (
        "mitigations",
        "high",
        (
            r"\bmitigation\s+mapping\b",
            r"\bmap\s+(?:threats?|risks?)\s+to\s+(?:controls?|mitigations?)\b",
            r"\bcontrols?\s+mapping\b",
            r"\bmitigations?\b",
        ),
    ),
    (
        "review_cadence",
        "medium",
        (
            r"\breview\s+cadence\b",
            r"\b(?:annual|quarterly|monthly)\s+threat\s+model\s+reviews?\b",
            r"\bthreat\s+model\s+reviews?\b",
            r"\breview\s+the\s+threat\s+model\b",
        ),
    ),
)


def detect_query_threat_modeling_requirements(query: str) -> dict[str, Any]:
    """Return threat-modeling requirements mentioned by a query."""
    text = _normalize_query(query)
    rows = _detect_rows(text) if _has_threat_modeling_context(text) else []
    return {
        "has_threat_modeling_requirements": bool(rows),
        "rows": rows,
    }


def _has_threat_modeling_context(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in _CONTEXT_PATTERNS)


def _detect_rows(text: str) -> list[dict[str, Any]]:
    rows: list[tuple[int, int, dict[str, Any]]] = []
    for index, (category, severity, patterns) in enumerate(_REQUIREMENTS):
        found = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if found:
            match = min(found, key=lambda item: item.start())
            rows.append(
                (
                    match.start(),
                    index,
                    {
                        "category": category,
                        "matched_text": match.group(0),
                        "severity": severity,
                    },
                )
            )
    return [row for _start, _index, row in sorted(rows, key=lambda item: (item[0], item[1]))]


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
