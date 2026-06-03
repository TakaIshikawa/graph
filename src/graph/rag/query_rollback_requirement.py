"""Detect rollback-related requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("rollback", "high", (r"\brollback\b", r"\broll\s+back\b", r"\bbackout\s+plan\b", r"\bback[-\s]out\s+procedure\b")),
    ("restore", "high", (r"\brestore\s+(?:from\s+)?backup\b", r"\bdata\s+restore\b", r"\brestore\s+(?:the\s+)?previous\s+version\b")),
    ("revert", "medium", (r"\brevert\s+(?:the\s+)?(?:deployment|release|production\s+change|operational\s+change)\b",)),
    ("migration_reversal", "high", (r"\bmigration\s+rollback\b", r"\broll\s+back\s+(?:the\s+)?migration\b", r"\breverse\s+(?:the\s+)?migration\b")),
    ("feature_flag_fallback", "medium", (r"\bfeature\s+flag\s+fallback\b", r"\bdisable\s+(?:the\s+)?feature\s+flag\b", r"\bturn\s+off\s+(?:the\s+)?feature\s+flag\b")),
    ("fallback", "medium", (r"\bfallback\s+plan\b", r"\bfallback\s+path\b", r"\bfailover\b", r"\bdegraded\s+mode\b")),
    ("disaster_recovery", "high", (r"\bdisaster\s+recovery\b", r"\bdr\s+plan\b", r"\brecovery\s+time\s+objective\b", r"\brecovery\s+point\s+objective\b")),
)
_VCS_CONTEXT = re.compile(r"\b(?:git|commit|branch|pull\s+request|merge\s+request|repository|repo|source[-\s]control|version[-\s]control)\b", re.I)
_OPERATIONAL_CONTEXT = re.compile(
    r"\b(?:deployment|release|production|service|migration|database|backup|restore|incident|outage|feature\s+flag|failover|disaster|recovery|data)\b",
    re.I,
)


def detect_query_rollback_requirements(query: str) -> list[dict[str, Any]]:
    """Return rollback, restore, fallback, and recovery requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    if not text or (_VCS_CONTEXT.search(text) and not _OPERATIONAL_CONTEXT.search(text)):
        return []

    rows: list[dict[str, Any]] = []
    for requirement, severity, patterns in _REQUIREMENTS:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append(
                {
                    "requirement": requirement,
                    "matched_text": match.group(0),
                    "severity": severity,
                    "span": [match.start(), match.end()],
                }
            )
    if any(row["requirement"] == "migration_reversal" for row in rows):
        rows = [
            row
            for row in rows
            if not (
                row["requirement"] == "rollback"
                and any(
                    other["requirement"] == "migration_reversal"
                    and other["span"][0] <= row["span"][0] < row["span"][1] <= other["span"][1]
                    for other in rows
                )
            )
        ]
    return sorted(rows, key=lambda row: (row["span"][0], row["span"][1], row["requirement"]))
