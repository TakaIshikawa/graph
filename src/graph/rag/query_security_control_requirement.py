"""Detect security-control requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("access_control", "high", (r"\baccess\s+controls?\b", r"\brbac\b", r"\brole[-\s]?based\s+access\b")),
    ("audit_logging", "high", (r"\baudit\s+logs?\b", r"\baudit\s+logging\b", r"\baudit\s+trails?\b")),
    ("encryption", "high", (r"\bencryption\b", r"\bencrypted\b", r"\bencrypt\b")),
    ("least_privilege", "high", (r"\bleast\s+privilege\b", r"\bminimum\s+privileges?\b")),
    ("mfa", "high", (r"\bmfa\b", r"\bmulti[-\s]?factor\s+auth(?:entication)?\b", r"\btwo[-\s]?factor\s+auth(?:entication)?\b")),
    ("secrets_handling", "high", (r"\bsecrets?\s+handling\b", r"\bsecret\s+management\b", r"\bapi\s+keys?\b", r"\bcredentials?\b")),
)


def detect_query_security_control_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if not text:
        return []

    rows: list[dict[str, Any]] = []
    for category, severity, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
