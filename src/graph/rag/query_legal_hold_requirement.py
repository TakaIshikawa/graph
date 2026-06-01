"""Detect legal hold requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUE_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("legal_hold", "high", (r"\blegal\s+hold\b", r"\blitigation\s+hold\b")),
    ("ediscovery", "high", (r"\be[-\s]?discovery\s+preservation\b", r"\bpreserve\s+for\s+e[-\s]?discovery\b")),
    ("retention_freeze", "high", (r"\bretention\s+freeze\b", r"\bfreeze\s+deletion\b", r"\bsuspend\s+deletion\b")),
    ("preserve_evidence", "medium", (r"\bpreserve\s+evidence\b", r"\bdo\s+not\s+delete\s+evidence\b")),
)


def detect_query_legal_hold_requirement(query: str) -> dict[str, Any]:
    """Return legal hold signals mentioned by a query."""
    text = " ".join(str(query or "").split())
    preservation_actions = [
        action for action, _severity, patterns in _CUE_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    severity = "high" if any(action in {"legal_hold", "ediscovery", "retention_freeze"} for action in preservation_actions) else "medium" if preservation_actions else "none"
    return {
        "requires_legal_hold": bool(preservation_actions),
        "preservation_actions": preservation_actions,
        "matched_cues": preservation_actions,
        "severity": severity,
    }
