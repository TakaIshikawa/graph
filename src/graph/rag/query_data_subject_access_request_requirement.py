"""Detect data subject access request requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("dsar", "high", (r"\bdsars?\b", r"\bdata\s+subject\s+access\s+requests?\b")),
    ("subject_access_request", "high", (r"\bsubject\s+access\s+requests?\b", r"\bprivacy\s+access\s+requests?\b")),
    ("access_request_workflow", "medium", (r"\baccess\s+request\s+workflows?\b", r"\bworkflow\s+for\s+(?:privacy\s+)?access\s+requests?\b")),
    ("identity_verification", "high", (r"\bidentity\s+verification\s+for\s+requests?\b", r"\bverify\s+request(?:er|or)\s+identity\b")),
    ("response_deadline", "high", (r"\bresponse\s+deadlines?\b", r"\brespond\s+(?:to\s+)?(?:dsars?|access\s+requests?)\s+within\b")),
    ("request_portal", "medium", (r"\brequest\s+portals?\b", r"\bdsar\s+portals?\b", r"\bprivacy\s+request\s+portals?\b")),
    ("fulfillment_evidence", "medium", (r"\bfulfillment\s+evidence\b", r"\bevidence\s+of\s+(?:dsar\s+)?fulfillment\b", r"\bproof\s+of\s+fulfillment\b")),
)

_PRIVACY_CONTEXT = re.compile(
    r"\b(?:dsars?|data\s+subject|subject\s+access|privacy\s+access|privacy\s+request|gdpr|ccpa|personal\s+data)\b",
    re.I,
)


def detect_query_data_subject_access_request_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if not text or not _PRIVACY_CONTEXT.search(text):
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
