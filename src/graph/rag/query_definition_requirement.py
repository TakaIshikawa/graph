"""Detect definition-oriented RAG query requirements."""

from __future__ import annotations

import re
from typing import Any

_CUE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("compare_terms", (r"\bdifference\s+between\b", r"\bdistinguish\b", r"\bcompare\s+\w+\s+(?:and|vs\.?|versus)\b")),
    ("definition", (r"\bwhat\s+is\b", r"\bdefine\b", r"\bdefinition\s+of\b", r"\bmeaning\s+of\b")),
    ("example_request", (r"\bexamples?\b", r"\bfor\s+example\b", r"\bsample\b")),
    ("taxonomy_request", (r"\btaxonomy\b", r"\bclassif(?:y|ication)\b", r"\btypes?\s+of\b", r"\bcategories\s+of\b")),
)


def detect_query_definition_requirement(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    cues = []
    labels = []
    for label, patterns in _CUE_SPECS:
        match = _first_match(normalized, patterns)
        if match:
            labels.append(label)
            cues.append({"label": label, "matched_text": match.group(0)})
    confidence = 0.0 if not labels else round(min(0.95, 0.55 + 0.12 * len(labels)), 2)
    return {
        "has_definition_requirement": bool(labels),
        "requirement_labels": sorted(labels),
        "matched_cues": sorted(cues, key=lambda row: (row["label"], row["matched_text"].casefold())),
        "confidence": confidence,
        "normalized_query": normalized,
    }


def _first_match(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
