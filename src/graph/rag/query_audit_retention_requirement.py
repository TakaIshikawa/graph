"""Detect audit-log retention requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_AUDIT_CONTEXT_RE = re.compile(r"\b(?:audit|logs?|audit[-\s]?trail|event\s+history)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("retention_duration", "high", (r"\bretain\s+(?:audit\s+)?logs?\s+for\b", r"\baudit[-\s]?log\s+retention\b", r"\blog\s+retention\s+(?:period|duration)\b")),
    ("immutable_storage", "high", (r"\bimmutable\b", r"\bworm\b", r"\bwrite\s+once\s+read\s+many\b", r"\btamper[-\s]?evident\b")),
    ("deletion_purge_policy", "medium", (r"\bdelet(?:e|ion)\s+(?:audit\s+)?logs?\b", r"\bpurge\s+(?:audit\s+)?logs?\b", r"\blog\s+purge\s+policy\b")),
    ("archival", "medium", (r"\barchiv(?:e|al|ing)\s+(?:audit\s+)?logs?\b", r"\bcold\s+storage\b")),
    ("access_review_evidence", "medium", (r"\baccess\s+review\s+evidence\b", r"\bevidence\s+for\s+access\s+reviews?\b")),
    ("compliance_retention", "high", (r"\bcompliance\s+retention\b", r"\bsox\s+retention\b", r"\bhipaa\s+retention\b", r"\bpci\s+retention\b")),
)


def detect_query_audit_retention_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    requirements = []
    if _AUDIT_CONTEXT_RE.search(text):
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_audit_retention_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
