"""Detect precision requirements expressed in a RAG query."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_CUES = {
    "exact": ("exact", "decimal", "significant figures", "nearest"),
    "range": ("range",),
    "approximate": ("approximately", "roughly", "estimate", "ballpark"),
}


def detect_query_precision_requirement(query: str) -> dict[str, Any]:
    normalized = " ".join((string(query) or "").casefold().split())
    matched = []
    for cues in _CUES.values():
        matched.extend(cue for cue in cues if cue in normalized)
    numeric = bool(re.search(r"\b\d+\s*(?:decimal places?|significant figures?|sig figs?)\b|\bnearest\s+\d+", normalized))
    level = "default"
    for candidate in ("exact", "range", "approximate"):
        if any(cue in matched for cue in _CUES[candidate]):
            level = candidate
            break
    confidence = 0.9 if numeric else 0.75 if matched else 0.2
    return {"normalized_query": normalized, "precision_level": level, "matched_cues": sorted(set(matched)), "numeric_precision_requested": numeric, "confidence": confidence}
