"""Detect backup, restore, and recovery requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("backup_policy", "medium", (r"\bbackup\s+polic(?:y|ies)\b", r"\bbackup\s+requirements?\b", r"\brequire\s+backups?\b")),
    ("restore", "high", (r"\brestore\s+(?:from\s+)?backups?\b", r"\bdata\s+restore\b", r"\brestore\s+procedures?\b")),
    ("rpo", "high", (r"\brpo\b", r"\brecovery\s+point\s+objective\b")),
    ("rto", "high", (r"\brto\b", r"\brecovery\s+time\s+objective\b")),
    ("point_in_time_restore", "high", (r"\bpoint[-\s]?in[-\s]?time\s+restore\b", r"\bpoint[-\s]?in[-\s]?time\s+recovery\b", r"\bpitr\b")),
    ("disaster_recovery", "high", (r"\bdisaster\s+recovery\b", r"\bdr\s+plan\b", r"\bdr\s+requirements?\b")),
    ("snapshot_retention", "medium", (r"\bsnapshot\s+retention\b", r"\bretain\s+snapshots?\b", r"\bsnapshots?\s+retained\b")),
)
_GENERIC_SAVE_EXPORT_RE = re.compile(r"\b(?:save|export)\s+(?:this|the)?\s*(?:file|csv|report|answer|results?|data)?\b", re.I)


def detect_backup_recovery_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if not matches:
            continue
        match = min(matches, key=lambda item: item.start())
        requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})

    requirements = sorted(requirements, key=lambda row: row["category"])
    if not requirements and _GENERIC_SAVE_EXPORT_RE.search(text):
        return {"has_backup_recovery_requirement": False, "requirements": []}
    return {"has_backup_recovery_requirement": bool(requirements), "requirements": requirements}
