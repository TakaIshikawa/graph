"""Audit whether next-step lines name an owner."""

from __future__ import annotations

import re
from typing import Any

_STEP_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)?(?:next|todo|action|follow\s+up|schedule|send|create|review|assign|confirm|draft|update|prepare|ask|meet|owner)\b", re.I)
_OWNER_LABEL_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?:")
_NON_OWNER_LABELS = {"action", "next", "todo", "owner"}
_OWNER_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("label", _OWNER_LABEL_RE),
    ("owner_field", re.compile(r"\bowner\s*=", re.I)),
    ("assigned_to", re.compile(r"\bassigned\s+to\s+[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)?\b", re.I)),
    ("team", re.compile(r"\b(?:engineering|product|design|sales|support|legal|finance|ops|operations|marketing)\s+team\b", re.I)),
)


def audit_answer_next_step_owner_coverage(answer: str) -> dict[str, Any]:
    """Return next-step owner coverage for action-oriented answers."""
    lines = [line.strip() for line in str(answer or "").splitlines() if line.strip()]
    steps = [line for line in lines if _is_step(line)]
    owner_cues: list[dict[str, Any]] = []
    missing: list[int] = []
    owned = 0
    for index, line in enumerate(steps):
        cues = _owner_cues(line, index)
        owner_cues.extend(cues)
        if cues:
            owned += 1
        else:
            missing.append(index)
    score = 1.0 if not steps else round(owned / len(steps), 3)
    return {
        "step_count": len(steps),
        "owned_step_count": owned,
        "missing_owner_indexes": missing,
        "owner_cues": owner_cues,
        "coverage_score": score,
    }


def _is_step(line: str) -> bool:
    return bool(_STEP_RE.search(line) or _OWNER_LABEL_RE.search(line))


def _owner_cues(line: str, step_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue_type, pattern in _OWNER_SPECS:
        for match in pattern.finditer(line):
            if cue_type == "label" and match.group(0).strip(" :-0123456789.").casefold() in _NON_OWNER_LABELS:
                continue
            rows.append({"step_index": step_index, "type": cue_type, "cue": match.group(0).strip(), "span": [match.start(), match.end()]})
    return rows
