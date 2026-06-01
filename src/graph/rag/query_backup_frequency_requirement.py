"""Detect backup-frequency requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("continuous", (r"\bcontinuous(?:ly)?\b", r"\bnear[-\s]?real[-\s]?time\s+backup\b")),
    ("daily", (r"\bdaily\b", r"\bevery\s+day\b")),
    ("hourly", (r"\bhourly\b", r"\bevery\s+hour\b")),
    ("point_in_time", (r"\bpoint[-\s]?in[-\s]?time\b", r"\bpitr\b")),
    ("retention_window", (r"\bretention\s+window\b", r"\bretain\s+backups?\s+for\b")),
    ("rpo", (r"\brpo\b", r"\brecovery\s+point\s+objective\b")),
    ("snapshot", (r"\bsnapshots?\b",)),
    ("weekly", (r"\bweekly\b", r"\bevery\s+week\b")),
)
_CADENCES = {"continuous", "daily", "hourly", "weekly"}


def detect_query_backup_frequency_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    signals = [name for name, patterns in _SIGNALS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    return {
        "requires_backup_frequency": bool(signals),
        "signals": signals,
        "cadence_terms": [name for name in signals if name in _CADENCES],
    }
