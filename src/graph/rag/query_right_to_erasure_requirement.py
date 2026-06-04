"""Detect right-to-erasure requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(
    r"\b(?:right\s+to\s+erasure|data\s+subject|privacy|gdpr|personal\s+data|deletion\s+request|erase\s+personal|delete\s+personal)\b",
    re.I,
)
_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("deletion_request", "high", (r"\bdeletion\s+requests?\b", r"\berasure\s+requests?\b", r"\bright\s+to\s+erasure\b")),
    ("identity_verification", "high", (r"\bidentity\s+verification\b", r"\bverify\s+(?:the\s+)?(?:requester|identity)\b")),
    ("retention_exception", "medium", (r"\bretention\s+exceptions?\b", r"\blegal\s+hold\b", r"\bretain(?:ed)?\s+for\s+legal\b")),
    ("processor_propagation", "high", (r"\bprocessor\s+propagation\b", r"\bpropagat(?:e|ion)\s+to\s+processors?\b", r"\bsubprocessors?\b")),
    ("backup_deletion", "medium", (r"\bbackup\s+deletion\b", r"\bdelete\s+from\s+backups?\b", r"\berase\s+backups?\b")),
    ("deadline_sla", "high", (r"\b(?:within|by|in)\s+\d+\s+(?:days?|hours?)\b", r"\bdeadline\b", r"\bsla\b")),
    ("audit_record", "medium", (r"\baudit\s+records?\b", r"\bdeletion\s+logs?\b", r"\berasure\s+logs?\b")),
)


def detect_query_right_to_erasure_requirement(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    if not _CONTEXT_RE.search(normalized):
        return {"has_right_to_erasure_requirement": False, "requirements": [], "normalized_query": normalized}

    requirements = []
    for category, severity, patterns in _CATEGORY_SPECS:
        match = _first_match(normalized, patterns)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})

    requirements.sort(key=lambda row: row["category"])
    return {
        "has_right_to_erasure_requirement": bool(requirements),
        "requirements": requirements,
        "normalized_query": normalized,
    }


def _first_match(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
