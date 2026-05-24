"""Classify requested answer granularity in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SPECS: tuple[tuple[str, int, re.Pattern[str]], ...] = (
    ("summary", 1, re.compile(r"\b(?:brief(?:ly)?|short|summary|summarize|high\s+level|tl;?dr)\b", re.I)),
    ("detailed", 2, re.compile(r"\b(?:in\s+detail|detailed|comprehensive|deep\s+dive|thorough(?:ly)?)\b", re.I)),
    ("stepwise", 4, re.compile(r"\b(?:step[-\s]?by[-\s]?step|steps?|procedure|walkthrough|sequence)\b", re.I)),
    ("itemized", 3, re.compile(r"\b(?:itemized|line\s+items?|bullet(?:ed)?|checklist|per\s+item)\b", re.I)),
    ("raw_data", 5, re.compile(r"\b(?:raw\s+(?:data|records?|rows?)|source\s+records?|verbatim|unaggregated)\b", re.I)),
)


def detect_query_granularity_requirement(query: str) -> dict[str, Any]:
    """Return the strongest detected requested granularity and evidence cues."""
    text = " ".join(str(query or "").split())
    cues: list[dict[str, Any]] = []
    for granularity, strength, pattern in _SPECS:
        for match in pattern.finditer(text):
            cues.append({"granularity": granularity, "cue": match.group(0).strip(), "span": [match.start(), match.end()], "strength": strength})
    cues.sort(key=lambda row: (row["span"][0], row["span"][1], row["granularity"]))
    strongest = "unknown"
    if cues:
        strongest = max(cues, key=lambda row: (row["strength"], -row["span"][0]))["granularity"]
    return {"granularity": strongest, "matched_cues": cues, "has_granularity_requirement": bool(cues)}
