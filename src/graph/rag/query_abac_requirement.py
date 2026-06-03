"""Detect attribute-based access control requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_ABAC_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"abac|"
    r"attribute[-\s]?based\s+(?:access\s+control|authorization)|"
    r"conditional\s+access(?:\s+polic(?:y|ies))?|"
    r"access\s+polic(?:y|ies)|"
    r"authorization\s+polic(?:y|ies)|"
    r"policy\s+decision\s+points?|"
    r"pdp"
    r")\b",
    re.I,
)

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "decision_point",
        "high",
        (
            r"\bpolicy\s+decision\s+points?\b",
            r"\bpdp\b",
            r"\bpolicy\s+engines?\b",
            r"\bauthorization\s+decision\s+points?\b",
        ),
    ),
    (
        "environment_attributes",
        "medium",
        (
            r"\benvironment(?:al)?\s+attributes?\b",
            r"\bcontext(?:ual)?\s+attributes?\b",
            r"\btime\s+of\s+day\b",
            r"\bnetwork\s+location\b",
            r"\bdevice\s+posture\b",
            r"\brisk\s+scores?\b",
        ),
    ),
    (
        "policy_conditions",
        "high",
        (
            r"\bconditional\s+access\b",
            r"\bpolicy\s+conditions?\b",
            r"\baccess\s+conditions?\b",
            r"\bauthorization\s+conditions?\b",
            r"\bcondition\s+expressions?\b",
            r"\bif\s+.*?\bthen\s+(?:allow|deny|permit|block)\b",
        ),
    ),
    (
        "resource_attributes",
        "medium",
        (
            r"\bresource\s+attributes?\b",
            r"\bobject\s+attributes?\b",
            r"\basset\s+attributes?\b",
            r"\bresource\s+(?:classification|sensitivity|owner|tenant)\b",
        ),
    ),
    (
        "subject_attributes",
        "medium",
        (
            r"\bsubject\s+attributes?\b",
            r"\buser\s+attributes?\b",
            r"\bprincipal\s+attributes?\b",
            r"\bidentity\s+attributes?\b",
            r"\buser\s+(?:department|clearance|role|group|location)\b",
        ),
    ),
)


def detect_query_abac_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    if not _ABAC_CONTEXT_RE.search(text):
        return {"has_abac_requirements": False, "requirements": []}

    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_abac_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
