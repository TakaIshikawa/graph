"""Detect backup retention and recovery requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "backup_retention",
        "medium",
        (
            r"\bbackup\s+retention\b",
            r"\bretain\s+backups?\b",
            r"\bkeep\s+backups?\s+for\b",
            r"\bbackups?\s+(?:kept|retained)\s+for\b",
            r"\bretention\s+(?:period|window)\b",
        ),
    ),
    (
        "point_in_time_recovery",
        "high",
        (r"\bpoint[-\s]?in[-\s]?time\s+recovery\b", r"\bpitr\b"),
    ),
    ("restore_testing", "high", (r"\brestore\s+test(?:ing|s)?\b", r"\btest(?:ed)?\s+restores?\b")),
    ("rpo", "high", (r"\brpo\b", r"\brecovery\s+point\s+objective\b")),
    ("rto", "high", (r"\brto\b", r"\brecovery\s+time\s+objective\b")),
    ("snapshot", "medium", (r"\bsnapshots?\b",)),
    (
        "backup_restore",
        "medium",
        (
            r"\bbackup\s*/\s*restore\b",
            r"\bbackup\s+and\s+restore\b",
            r"\brestore\s+guarantee\b",
            r"\brestore\s+(?:sla|slo|objective|requirement)\b",
        ),
    ),
)
_REQUIREMENT_CONTEXT_RE = re.compile(
    r"\b(require|requires|required|requirement|must|need|needs|policy|sla|slo|guarantee|guaranteed|objective|window|period)\b",
    re.I,
)
_CASUAL_BACKUP_RE = re.compile(r"\bbackup\s+(?:file|copy|notes?|doc(?:ument)?|folder|spreadsheet|csv|export)\b", re.I)


def detect_query_backup_retention_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if not matches:
            continue
        match = min(matches, key=lambda item: item.start())
        if category == "backup_retention" and _is_casual_backup_mention(text):
            continue
        requirements.append(
            {
                "category": category,
                "matched_text": match.group(0),
                "severity": _severity(category, severity, text),
            }
        )

    requirements = sorted(requirements, key=lambda row: row["category"])
    return {
        "has_backup_retention_requirement": bool(requirements),
        "requirements": requirements,
    }


def _is_casual_backup_mention(text: str) -> bool:
    return bool(_CASUAL_BACKUP_RE.search(text)) and not _REQUIREMENT_CONTEXT_RE.search(text)


def _severity(category: str, default: str, text: str) -> str:
    if category in {"rpo", "rto", "point_in_time_recovery", "restore_testing"}:
        return "high"
    if category == "backup_restore" and re.search(r"\b(guarantee|guaranteed|sla|slo|objective|required|must)\b", text, re.I):
        return "high"
    return default
