"""Detect graceful shutdown requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("graceful_shutdown", (r"\bgraceful\s+shutdown\b", r"\bshutdown\s+gracefully\b")),
    ("connection_drain", (r"\bdrain\s+connections?\b", r"\bconnection\s+drain(?:ing)?\b")),
    ("termination_grace_period", (r"\btermination\s+grace\s+period\b", r"\bgrace\s+period\s+for\s+termination\b")),
    ("inflight_completion", (r"\bin[-\s]?flight\s+(?:request\s+)?completion\b", r"\bcomplete\s+in[-\s]?flight\s+requests?\b")),
    ("sigterm_handling", (r"\bSIGTERM\b", r"\bhandle\s+termination\s+signals?\b")),
    ("shutdown_hook", (r"\bshutdown\s+hooks?\b", r"\bpre[-\s]?stop\s+hooks?\b")),
)


def detect_query_graceful_shutdown_requirement(query: str) -> dict[str, Any]:
    """Return graceful shutdown requirement categories mentioned by a query."""
    requirements = _requirements(query)
    return {"has_graceful_shutdown_requirement": bool(requirements), "requirements": requirements}


def _requirements(query: str) -> list[dict[str, str]]:
    text = " ".join(str(query or "").split())
    rows: list[dict[str, str]] = []
    for category, patterns in _CATEGORIES:
        match = _first_match(patterns, text)
        if match:
            rows.append({"category": category, "matched_text": match.group(0)})
    return sorted(rows, key=lambda row: row["category"])


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None
