"""Classify retrieved RAG snippets by actionability."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence, Mapping
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_IMPERATIVE_RE = re.compile(r"(?i)\b(?:do|run|create|add|set|check|verify|use|choose|compare|install|open|write|review|measure|calculate|record)\b")
_CHECKLIST_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)]|\[[ xX]\])\s+")
_PROCEDURE_RE = re.compile(r"(?i)\b(?:step \d+|first,|next,|then,|finally,|procedure|workflow|instructions|how to)\b")
_COMMAND_RE = re.compile(r"(?m)(?:^|\s)(?:python|pytest|uv|pip|npm|git|curl|docker|kubectl)\s+[\w./:-]+")
_EXAMPLE_RE = re.compile(r"(?i)\b(?:for example|example:|e\.g\.|sample)\b")
_DECISION_RE = re.compile(r"(?i)\b(?:if|when|unless|choose|prefer|criteria|tradeoff|threshold|decision)\b")
_EXPLANATORY_RE = re.compile(r"(?i)\b(?:means|refers to|because|overview|background|definition|explains|describes)\b")
_REFERENCE_RE = re.compile(r"(?i)\b(?:api reference|schema|field|parameter|table|appendix|specification|version|license)\b")

_ACTION_CUES = {
    "imperative_verb": _IMPERATIVE_RE,
    "checklist": _CHECKLIST_RE,
    "procedure": _PROCEDURE_RE,
    "command": _COMMAND_RE,
    "example": _EXAMPLE_RE,
    "decision_criteria": _DECISION_RE,
}


def _label(cues: list[str], text: str) -> tuple[str, float]:
    if {"imperative_verb", "procedure"} & set(cues) or len(cues) >= 3:
        return "actionable", min(0.95, 0.55 + len(cues) * 0.12)
    if _REFERENCE_RE.search(text):
        return "reference-only", 0.72
    if _EXPLANATORY_RE.search(text):
        return "explanatory", 0.68
    if cues:
        return "explanatory", 0.58
    return "low-action", 0.5


def classify_result_actionability(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return per-result actionability labels and aggregate counts."""
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, result in enumerate(results):
        text = content_text(result)
        cues = [name for name, pattern in _ACTION_CUES.items() if pattern.search(text)]
        label, confidence = _label(cues, text)
        counts[label] += 1
        rows.append(
            {
                "result_id": result_id(result, index),
                "label": label,
                "confidence": round(confidence, 2),
                "matched_cues": cues,
            }
        )
    return {
        "result_count": len(rows),
        "label_counts": {label: counts.get(label, 0) for label in ("actionable", "explanatory", "reference-only", "low-action")},
        "results": rows,
    }
