"""Detect verification depth requested by a RAG query."""

from __future__ import annotations

import re
from typing import Any

_SHALLOW_CUES = ("quick check", "sanity check", "quick sanity check", "spot check")
_DEEP_CUES = ("source every claim", "audit", "prove", "verify", "double-check", "double check", "cross-check", "cross check")


def detect_query_verification_depth_requirement(query: str) -> dict[str, Any]:
    """Classify a query as requiring shallow, normal, or deep verification."""
    normalized_query = _normalize(query)
    matched_cues = _matched_cues(normalized_query)
    shallow = [cue for cue in matched_cues if cue["depth"] == "shallow"]
    deep = [cue for cue in matched_cues if cue["depth"] == "deep"]
    if deep:
        required_depth = "deep"
        confidence = 0.9 if any(cue["cue"] in {"source every claim", "audit", "prove"} for cue in deep) else 0.8
        suggested_retrieval_passes = 3
    elif shallow:
        required_depth = "shallow"
        confidence = 0.75
        suggested_retrieval_passes = 1
    else:
        required_depth = "normal"
        confidence = 0.35
        suggested_retrieval_passes = 2
    return {
        "normalized_query": normalized_query,
        "required_depth": required_depth,
        "confidence": confidence,
        "matched_cues": matched_cues,
        "suggested_retrieval_passes": suggested_retrieval_passes,
    }


def _matched_cues(normalized_query: str) -> list[dict[str, Any]]:
    rows = []
    for depth, cues in (("shallow", _SHALLOW_CUES), ("deep", _DEEP_CUES)):
        for cue in cues:
            match = re.search(rf"\b{re.escape(cue)}\b", normalized_query)
            if match:
                rows.append({"cue": cue, "depth": depth, "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["cue"]))
    return rows


def _normalize(query: str) -> str:
    return " ".join(str(query or "").casefold().split())
