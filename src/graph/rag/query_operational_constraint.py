"""Detect operational constraints in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "no_downtime",
        re.compile(
            r"\b(?:no\s+downtime|zero\s+downtime|without\s+downtime|avoid\s+downtime|no\s+service\s+interruption|without\s+service\s+interruption|keep\s+(?:it|the\s+system|service)\s+running)\b",
            re.I,
        ),
    ),
    (
        "limited_staff",
        re.compile(
            r"\b(?:limited\s+staff|small\s+team|skeleton\s+crew|short[-\s]staffed|only\s+\d+\s+(?:people|engineers|staff|operators)|minimal\s+(?:staff|headcount|ops\s+support))\b",
            re.I,
        ),
    ),
    (
        "offline_mode",
        re.compile(
            r"\b(?:offline\s+mode|works?\s+offline|without\s+(?:internet|network|connectivity)|air[-\s]gapped|disconnected\s+(?:mode|environment)|local[-\s]only)\b",
            re.I,
        ),
    ),
    (
        "migration_window",
        re.compile(
            r"\b(?:migration\s+window|during\s+(?:the\s+)?migration|cutover\s+window|data\s+migration\s+window|migration\s+period)\b",
            re.I,
        ),
    ),
    (
        "maintenance_window",
        re.compile(
            r"\b(?:maintenance\s+window|scheduled\s+maintenance|during\s+maintenance|maintenance\s+period|change\s+window)\b",
            re.I,
        ),
    ),
    (
        "rollback_requirement",
        re.compile(
            r"\b(?:rollback\s+(?:plan|required|requirement|strategy)|roll\s+back|revert\s+(?:plan|path|quickly)?|backout\s+(?:plan|procedure)|back\s+out\s+(?:plan|procedure)?)\b",
            re.I,
        ),
    ),
    (
        "dependency_freeze",
        re.compile(
            r"\b(?:(?:dependency|dependencies|package|packages|library|libraries)\s+freeze|freeze\s+(?:dependencies|packages|libraries)|no\s+(?:new|additional)\s+(?:dependencies|packages|libraries)|cannot\s+(?:add|upgrade)\s+(?:dependencies|packages|libraries)|without\s+(?:new|additional)\s+(?:dependencies|packages|libraries))\b",
            re.I,
        ),
    ),
)


def detect_query_operational_constraint(query: str) -> dict[str, Any]:
    """Return operational constraint types and matched evidence cues."""
    text = " ".join(str(query or "").split())
    cues: list[dict[str, Any]] = []
    for kind, pattern in _SPECS:
        for match in pattern.finditer(text):
            cues.append({"type": kind, "cue": match.group(0).strip(), "span": [match.start(), match.end()]})
    cues.sort(key=lambda row: (row["span"][0], row["span"][1], row["type"]))
    return {
        "has_operational_constraints": bool(cues),
        "constraint_types": sorted({row["type"] for row in cues}),
        "matched_cues": cues,
    }
