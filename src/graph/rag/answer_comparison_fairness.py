"""Audit fairness of comparison-style RAG answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text

_COMPARATOR_RE = re.compile(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)?)\s+(?:vs\.?|versus|compared\s+(?:with|to)|against)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)?)\b")
_CRITERIA_RE = re.compile(r"\b(?:criteria|based\s+on|using|measured\s+by|cost|quality|speed|accuracy|risk|latency|price)\b", re.I)
_WINNER_RE = re.compile(r"\b(?:best|better|wins?|winner|superior|outperforms|dominates)\b", re.I)


def audit_answer_comparison_fairness(answer: str, evidence: Iterable[Any] | None = None) -> dict[str, Any]:
    """Return fairness warnings for comparative answers."""
    text = " ".join(str(answer or "").split())
    evidence_text = " ".join(content_text(item) for item in (evidence or []))
    comparators = _comparators(text)
    criteria = _CRITERIA_RE.findall(text)
    winner = bool(_WINNER_RE.search(text))
    mentions = {name: len(re.findall(rf"\b{re.escape(name)}\b", text, re.I)) for name in comparators}
    evidence_mentions = {name: len(re.findall(rf"\b{re.escape(name)}\b", evidence_text, re.I)) for name in comparators}
    warnings: list[str] = []
    if len(comparators) >= 2 and not criteria:
        warnings.append("missing_comparator_criteria")
    if len(comparators) >= 2 and max(mentions.values(), default=0) > min(mentions.values(), default=0) + 2:
        warnings.append("one_sided_comparison")
    if winner and (not evidence_text or any(count == 0 for count in evidence_mentions.values())):
        warnings.append("unsupported_winner_language")
    score = max(0.0, 1.0 - 0.3 * len(warnings))
    return {
        "fairness_score": round(score, 2),
        "warnings": warnings,
        "comparators": comparators,
        "matched_cues": {"criteria": sorted(set(criteria), key=str.casefold), "winner_language": winner},
    }


def _comparators(text: str) -> list[str]:
    names: list[str] = []
    for match in _COMPARATOR_RE.finditer(text):
        for group in (1, 2):
            name = match.group(group).strip()
            if name not in names:
                names.append(name)
    return names
