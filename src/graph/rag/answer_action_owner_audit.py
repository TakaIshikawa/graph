"""Audit whether action-like answer lines include an owner."""

from __future__ import annotations

import re
from typing import Any

_ACTION_RE = re.compile(
    r"^\s*(?:[-*]\s+\[[ xX]\]\s+|[-*]\s+|\d+[.)]\s+)?(?:action|todo|next|follow\s+up|send|create|review|assign|confirm|draft|update|prepare|schedule|ask|meet|verify|document|share)\b",
    re.IGNORECASE,
)
_OWNER_LABEL_RE = re.compile(r"\b(?:owner|assignee|responsible)\s*[:=]\s*[^,;.]+", re.IGNORECASE)
_ASSIGNED_TO_RE = re.compile(r"\bassigned\s+to\s+[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)?\b", re.IGNORECASE)
_PERSON_LABEL_RE = re.compile(r"^\s*(?:[-*]\s+\[[ xX]\]\s+|[-*]\s+|\d+[.)]\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?:")
_TEAM_RE = re.compile(r"\b(?:engineering|product|design|sales|support|legal|finance|ops|operations|marketing|research|data)\s+team\b", re.IGNORECASE)


def audit_answer_action_owners(answer: str) -> dict[str, Any]:
    actions = [line.strip() for line in str(answer or "").splitlines() if line.strip() and _is_action(line)]
    missing = [line for line in actions if not _has_owner(line)]
    action_count = len(actions)
    owner_coverage_ratio = round((action_count - len(missing)) / action_count, 3) if action_count else 1.0
    warnings = []
    if missing:
        warnings.append("actions_missing_owner")
    return {
        "action_count": action_count,
        "actions_missing_owner": len(missing),
        "owner_coverage_ratio": owner_coverage_ratio,
        "sampled_actions": missing[:3],
        "warnings": warnings,
    }


def _is_action(line: str) -> bool:
    return bool(_ACTION_RE.search(line) or _PERSON_LABEL_RE.search(line))


def _has_owner(line: str) -> bool:
    return bool(_OWNER_LABEL_RE.search(line) or _ASSIGNED_TO_RE.search(line) or _PERSON_LABEL_RE.search(line) or _TEAM_RE.search(line))
