"""Classify method signals in RAG evidence items."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, string, value

_METHODS: dict[str, tuple[re.Pattern[str], ...]] = {
    "survey": (re.compile(r"\bsurveys?\b", re.I),),
    "randomized": (re.compile(r"\brandomi[sz]ed\b", re.I), re.compile(r"\bRCT\b", re.I)),
    "interview": (re.compile(r"\binterviews?\b", re.I),),
    "case_study": (re.compile(r"\bcase\s+stud(?:y|ies)\b", re.I),),
    "benchmark": (re.compile(r"\bbenchmarks?\b", re.I),),
    "review": (re.compile(r"\breviews?\b", re.I), re.compile(r"\bmeta-analysis\b", re.I)),
    "experiment": (re.compile(r"\bexperiments?\b", re.I),),
}


def classify_evidence_method_signals(evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Return method counts and item-level cue matches."""
    counts = {method: 0 for method in _METHODS}
    matched_items = []
    for index, item in enumerate(evidence_items or []):
        text = _text(item)
        for method, patterns in _METHODS.items():
            match = next((pattern.search(text) for pattern in patterns if pattern.search(text)), None)
            if match:
                counts[method] += 1
                matched_items.append({"item_id": result_id(item, index), "method": method, "matched_cue": match.group(0)})
    return {"method_counts": counts, "matched_items": matched_items}


def _text(item: Any) -> str:
    parts = [content_text(item)]
    for key in ("method", "methods", "study_type"):
        text = string(value(item, key))
        if text:
            parts.append(text)
    return " ".join(parts)
