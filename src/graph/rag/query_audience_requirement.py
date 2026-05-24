"""Detect requested audience requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_AUDIENCES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("beginner", "plain", "introductory", (r"\bbeginners?\b", r"\bnon[- ]?expert\b", r"\blay(?:person|people| audience)?\b", r"\bplain english\b")),
    ("expert", "technical", "advanced", (r"\bexperts?\b", r"\badvanced\b", r"\bspecialists?\b", r"\btechnical audience\b")),
    ("executive", "concise", "strategic_summary", (r"\bexecutives?\b", r"\bC[- ]?suite\b", r"\bleadership\b", r"\bboard\b")),
    ("clinician", "clinical", "clinical_detail", (r"\bclinicians?\b", r"\bdoctors?\b", r"\bphysicians?\b", r"\bnurses?\b")),
    ("engineer", "technical", "implementation_detail", (r"\bengineers?\b", r"\bdevelopers?\b", r"\bdevops\b", r"\barchitects?\b")),
    ("policymaker", "policy", "policy_brief", (r"\bpolicy ?makers?\b", r"\bregulators?\b", r"\blegislators?\b", r"\bgovernment officials?\b")),
    ("student", "plain", "teaching", (r"\bstudents?\b", r"\bchildren\b", r"\bkids\b", r"\bhigh school\b", r"\bmiddle school\b")),
)


def detect_query_audience_requirement(query: str) -> dict[str, Any]:
    """Return normalized audience requirements and explanation-depth hints."""
    normalized = " ".join(str(query or "").split())
    requirements = []
    for audience, register, depth, patterns in _AUDIENCES:
        phrases = _phrases(normalized, patterns)
        if phrases:
            requirements.append(
                {
                    "audience": audience,
                    "matched_phrases": phrases,
                    "communication_register": register,
                    "suggested_explanation_depth": depth,
                    "confidence": 0.85,
                }
            )
    return {
        "query": normalized,
        "requires_audience_adaptation": bool(requirements),
        "audiences": [row["audience"] for row in requirements],
        "requirements": requirements,
        "matched_phrases": [phrase for row in requirements for phrase in row["matched_phrases"]],
    }


def _phrases(query: str, patterns: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, query, re.I):
            phrase = match.group(0).strip()
            key = phrase.casefold()
            if key not in seen:
                seen.add(key)
                found.append(phrase)
    return found
