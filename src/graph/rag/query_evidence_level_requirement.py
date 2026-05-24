"""Detect evidence-level requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("systematic_review", "synthesis", "highest", (r"\bsystematic reviews?\b", r"\bmeta[- ]analys(?:is|es)\b")),
    ("randomized_trial", "experimental", "high", (r"\brandomi[sz]ed (?:controlled )?trials?\b", r"\bRCTs?\b")),
    ("peer_reviewed", "scholarly", "high", (r"\bpeer[- ]reviewed\b", r"\bscholarly articles?\b")),
    ("official_source", "official", "high", (r"\bofficial (?:documentation|docs|sources?|guidance)\b", r"\bgovernment sources?\b")),
    ("primary_source", "primary", "medium", (r"\bprimary sources?\b", r"\boriginal documents?\b", r"\bfirst[- ]hand\b")),
    ("expert_opinion", "expert", "medium", (r"\bexpert opinions?\b", r"\bexpert commentary\b", r"\bspecialist opinions?\b")),
    ("anecdotal_evidence", "anecdotal", "low", (r"\banecdotal evidence\b", r"\bcase reports?\b", r"\bpersonal experiences?\b")),
)


def detect_query_evidence_level_requirement(query: str) -> dict[str, Any]:
    """Return structured evidence-level requirements requested by a query."""
    normalized = " ".join(str(query or "").split())
    requirements = []
    for label, category, tier, patterns in _REQUIREMENTS:
        for phrase, start, end in _matches(normalized, patterns):
            requirements.append(
                {
                    "requirement": label,
                    "matched_phrase": phrase,
                    "span": (start, end),
                    "evidence_category": category,
                    "evidence_tier": tier,
                    "confidence": 0.88,
                }
            )
    requirements.sort(key=lambda row: (row["span"][0], row["requirement"]))
    return {
        "query": normalized,
        "requires_evidence_level": bool(requirements),
        "requirements": requirements,
        "matched_phrases": [row["matched_phrase"] for row in requirements],
        "requirement_labels": [row["requirement"] for row in requirements],
    }


def _matches(query: str, patterns: tuple[str, ...]) -> list[tuple[str, int, int]]:
    rows = []
    seen: set[tuple[int, int]] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, query, re.I):
            span = match.span()
            if span not in seen:
                seen.add(span)
                rows.append((match.group(0).strip(), span[0], span[1]))
    return rows
