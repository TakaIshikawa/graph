"""Prioritize context gaps for follow-up retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import string

_SEVERITY = {"critical": 40, "high": 30, "medium": 20, "low": 10}
_ACTIONS = {
    "date": "retrieve current dated sources",
    "recency": "retrieve current dated sources",
    "source": "retrieve authoritative source records",
    "entity": "retrieve entity-specific context",
    "definition": "retrieve definitional reference material",
}


def prioritize_context_gaps(gaps: Iterable[Mapping[str, Any]], query_intent: str | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Rank context gaps by severity, likely impact, recency relevance, and fillability."""
    intent_text = string(query_intent) if not isinstance(query_intent, Mapping) else string(query_intent.get("intent") or query_intent.get("query"))
    rows = []
    for index, gap in enumerate(gaps or []):
        gap_type = string(gap.get("gap_type") or gap.get("type")) or "unknown"
        severity = string(gap.get("severity")) or "medium"
        impact = string(gap.get("answer_impact") or gap.get("impact")) or "medium"
        fillable = gap.get("fillable", True)
        score = _SEVERITY.get(severity, 20) + _SEVERITY.get(impact, 20) // 2 + (10 if fillable else 0)
        if intent_text and any(token in intent_text.casefold() for token in ("latest", "current", "policy", "recommend")):
            score += 10
        action = _action(gap_type)
        rows.append(
            {
                **dict(gap),
                "gap_type": gap_type,
                "priority_score": score,
                "priority": "high" if score >= 55 else ("medium" if score >= 35 else "low"),
                "priority_reasons": _reasons(severity, impact, bool(fillable), intent_text),
                "suggested_retrieval_action": action,
                "_index": index,
            }
        )
    rows.sort(key=lambda row: (-int(row["priority_score"]), row["_index"]))
    for row in rows:
        row.pop("_index", None)
    return {"prioritized_gaps": rows, "gap_count": len(rows)}


def _action(gap_type: str) -> str:
    lowered = gap_type.casefold()
    for key, action in _ACTIONS.items():
        if key in lowered:
            return action
    return "retrieve targeted supporting context"


def _reasons(severity: str, impact: str, fillable: bool, intent: str | None) -> list[str]:
    reasons = [f"{severity}_severity", f"{impact}_answer_impact"]
    if fillable:
        reasons.append("fillable")
    if intent:
        reasons.append("query_intent_relevant")
    return reasons
