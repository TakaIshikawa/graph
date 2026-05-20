"""Classify query wording by the shape of evidence likely needed."""

from __future__ import annotations

import re
from typing import Any

_SHAPES = [
    ("comparison", 4, ("compare", "versus", "vs", "difference", "better", "pros and cons")),
    ("timeline", 5, ("timeline", "history", "over time", "when did", "chronology", "before", "after")),
    ("diagnostic", 4, ("why", "cause", "root cause", "diagnose", "troubleshoot", "error", "fix")),
    ("how_to", 3, ("how to", "steps", "guide", "tutorial", "implement", "setup", "configure")),
    ("opinion_or_preference", 3, ("should i", "best", "recommend", "prefer", "worth it", "opinion")),
    ("fact_lookup", 1, ("what is", "who is", "where is", "define", "date of", "number of")),
]


def classify_query_evidence_shape(query: str, *, result_count: int | None = None) -> dict[str, Any]:
    """Return an evidence-shape label, matched cues, and sufficiency warnings."""
    text = " ".join(str(query).strip().split())
    if not text:
        raise ValueError("query must not be blank")
    if result_count is not None and (not isinstance(result_count, int) or isinstance(result_count, bool) or result_count < 0):
        raise ValueError("result_count must be a non-negative integer or None")

    lowered = text.casefold()
    matches: list[tuple[str, int, list[str]]] = []
    for shape, minimum, cues in _SHAPES:
        found = [cue for cue in cues if _contains(lowered, cue)]
        if found:
            matches.append((shape, minimum, found))
    if not matches:
        matches = [("fact_lookup", 1, [])]

    shape, minimum, cues = sorted(matches, key=lambda item: (-len(item[2]), -item[1], item[0]))[0]
    warnings = []
    if result_count is not None and result_count < minimum:
        warnings.append("insufficient_result_count")
    return {
        "shape": shape,
        "matched_cues": cues,
        "recommended_min_results": minimum,
        "warnings": warnings,
    }


def _contains(text: str, cue: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text))
