"""Detect audit logging requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("access_logs", "high", (r"\baccess\s+logs?\b", r"\blog\s+access\b")),
    ("audit_logs", "high", (r"\baudit\s+logs?\b", r"\bauditability\b")),
    ("evidence_trail", "medium", (r"\bevidence\s+trails?\b", r"\baudit\s+trails?\b")),
    ("immutable_logs", "high", (r"\bimmutable\s+logs?\b", r"\btamper[-\s]?proof\s+logs?\b")),
    ("traceability", "high", (r"\btraceability\b", r"\btraceable\b")),
    ("who_did_what", "high", (r"\bwho[-\s]?did[-\s]?what\b", r"\bwho\s+changed\s+what\b")),
)


def detect_query_audit_logging_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
