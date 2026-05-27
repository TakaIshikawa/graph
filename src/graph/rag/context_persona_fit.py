"""Audit whether retrieved context fits the query persona."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, string, value

_PERSONAS: dict[str, tuple[str, ...]] = {
    "beginner": ("beginner", "intro", "simple", "plain english", "101"),
    "expert": ("expert", "advanced", "technical", "deep dive"),
    "executive": ("executive", "executives", "leadership", "board", "strategy"),
    "engineer": ("engineer", "engineers", "developer", "developers", "api", "architecture"),
    "clinician": ("clinician", "clinical", "patient", "medical"),
    "student": ("student", "classroom", "homework", "learn"),
    "legal": ("legal", "lawyer", "regulation", "compliance"),
    "financial": ("financial", "investor", "finance", "accounting"),
    "consumer": ("consumer", "buyer", "customer", "personal"),
}


def audit_context_persona_fit(query: str, context_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Return persona fit counts and mismatched context items."""
    personas = _infer_personas(query)
    counts = {"matching": 0, "mismatching": 0, "neutral": 0}
    mismatches = []
    for index, item in enumerate(context_items or []):
        item_personas = _infer_personas(_item_text(item))
        if not personas or not item_personas:
            counts["neutral"] += 1
        elif personas & item_personas:
            counts["matching"] += 1
        else:
            counts["mismatching"] += 1
            mismatches.append(
                {
                    "item_id": result_id(item, index),
                    "expected_personas": sorted(personas),
                    "matched_personas": sorted(item_personas),
                    "reason": "context audience conflicts with query persona",
                }
            )
    return {
        "inferred_personas": sorted(personas),
        "fit_counts": counts,
        "mismatched_items": mismatches,
        "has_persona_mismatch_risk": bool(mismatches),
    }


def _infer_personas(text: Any) -> set[str]:
    normalized = (string(text) or "").casefold()
    return {persona for persona, cues in _PERSONAS.items() if any(re.search(rf"\b{re.escape(cue)}\b", normalized) for cue in cues)}


def _item_text(item: Any) -> str:
    parts = [content_text(item)]
    for key in ("audience", "source_type"):
        text = string(value(item, key))
        if text:
            parts.append(text)
    return " ".join(parts)
