"""Heuristic audit for answer actionability."""

from __future__ import annotations

import re
from typing import Any

_STEP_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*(.+)$", re.MULTILINE)
_IMPERATIVE_RE = re.compile(r"^(?:create|update|send|review|assign|schedule|measure|document|verify|run|contact)\b", re.IGNORECASE)
_DEADLINE_RE = re.compile(r"\b(?:by|before|due)\s+([A-Z][a-z]+ \d{1,2}|\d{4}-\d{2}-\d{2}|tomorrow|next week)\b", re.IGNORECASE)
_OWNER_RE = re.compile(r"\b(?:owner|assignee|assigned to|by)\s*:?\s*([A-Z][A-Za-z0-9 _-]{1,40})")
_PREREQ_RE = re.compile(r"\b(?:requires?|prerequisite|depends on|after)\s+([^.;\n]+)", re.IGNORECASE)
_VAGUE_RE = re.compile(r"\b(?:improve|optimize|handle|fix|address|leverage)\b(?:\s*(?:it|things|stuff|this|that))?(?:[.;\n]|$)", re.IGNORECASE)


def audit_answer_actionability(answer: str, query_intent: str | None = None) -> dict[str, Any]:
    text = str(answer)
    candidates = [match.group(1).strip() for match in _STEP_RE.finditer(text)]
    if not candidates:
        candidates = [line.strip() for line in text.splitlines() if _IMPERATIVE_RE.search(line.strip())]
    next_steps = [step for step in candidates if step]
    owners = sorted({match.group(1).strip() for match in _OWNER_RE.finditer(text)})
    deadlines = sorted({match.group(1).strip() for match in _DEADLINE_RE.finditer(text)})
    prerequisites = sorted({match.group(1).strip() for match in _PREREQ_RE.finditer(text)})
    vague_flags = sorted({match.group(0).strip(" .;\n") for match in _VAGUE_RE.finditer(text)})
    score = min(1.0, 0.2 + 0.15 * len(next_steps) + 0.15 * bool(owners) + 0.15 * bool(deadlines) + 0.1 * bool(prerequisites) - 0.1 * len(vague_flags))
    if query_intent and "action" in query_intent.casefold():
        score = min(1.0, score + 0.1)
    return {
        "actionability_score": round(max(0.0, score), 2),
        "next_steps": next_steps,
        "owners": owners,
        "deadlines": deadlines,
        "prerequisites": prerequisites,
        "vague_action_flags": vague_flags,
    }
