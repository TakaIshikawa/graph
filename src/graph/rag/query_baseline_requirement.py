"""Detect historical baseline requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("previous_period", re.compile(r"\b(?:compared\s+(?:with|to)|versus|vs\.?|relative\s+to)\s+(last\s+(?:year|quarter|month|week)|previous\s+(?:period|year|quarter|month))\b", re.I)),
    ("before_after", re.compile(r"\b(before\s+vs\.?\s+after|pre[-\s]launch|post[-\s]launch|before\s+and\s+after)\b", re.I)),
    ("baseline", re.compile(r"\b(?:relative\s+to|against|compared\s+(?:with|to))\s+(?:the\s+)?(baseline|current\s+state|control|pre[-\s]\w+)\b", re.I)),
)


def detect_query_baseline_requirement(query: str) -> dict[str, Any]:
    """Return baseline comparison cues and anchors."""
    text = " ".join(str(query or "").split())
    cues: list[dict[str, Any]] = []
    anchors: list[str] = []
    for kind, pattern in _SPECS:
        for match in pattern.finditer(text):
            cues.append({"type": kind, "cue": match.group(0).strip(), "span": [match.start(), match.end()]})
            if match.lastindex and kind != "before_after":
                anchors.append(match.group(match.lastindex).strip())
    cues.sort(key=lambda row: (row["span"][0], row["span"][1], row["type"]))
    return {"requires_baseline": bool(cues), "matched_cues": cues, "baseline_anchors": sorted(dict.fromkeys(anchors), key=str.casefold)}
