"""Detect threshold and tolerance language in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lower_bound", re.compile(r"\b(?:at\s+least|no\s+less\s+than|above|over|>=|>)\s*(?:p\d{1,2}|\$?\d+(?:\.\d+)?\s*(?:%|percent|ms|s|seconds?|days?)?)(?=\s|[.?!;,]|$)", re.I)),
    ("upper_bound", re.compile(r"\b(?:at\s+most|no\s+more\s+than|under|below|less\s+than|<=|<)\s*(?:p\d{1,2}|\$?\d+(?:\.\d+)?\s*(?:%|percent|ms|s|seconds?|days?)?)(?=\s|[.?!;,]|$)", re.I)),
    ("range", re.compile(r"\bwithin\s+(?:tolerance|range|(?:\+/-|±)?\s*\d+(?:\.\d+)?\s*(?:%|percent|ms|s|seconds?|days?)?)(?=\s|[.?!;,]|$)", re.I)),
    ("percentile", re.compile(r"\b(?:p\d{1,2}|percentile)\b", re.I)),
    ("materiality", re.compile(r"\b(?:material\s+change|materiality|significant\s+change|meaningful\s+difference)\b", re.I)),
)
_VALUE_RE = re.compile(r"(?:p\d{1,2}|\$?\d+(?:\.\d+)?\s*(?:%|percent|ms|s|seconds?|days?)?)(?=\s|[.?!;,]|$)", re.I)


def detect_query_sensitivity_threshold(query: str) -> dict[str, Any]:
    """Return threshold cues and extracted numeric values."""
    text = " ".join(str(query or "").split())
    cues: list[dict[str, Any]] = []
    thresholds: list[str] = []
    for kind, pattern in _SPECS:
        for match in pattern.finditer(text):
            cue = match.group(0).strip()
            cues.append({"type": kind, "cue": cue, "span": [match.start(), match.end()]})
            thresholds.extend(value.group(0).strip() for value in _VALUE_RE.finditer(cue))
    cues.sort(key=lambda row: (row["span"][0], row["span"][1], row["type"]))
    return {
        "has_threshold": bool(cues),
        "threshold_types": sorted({row["type"] for row in cues}),
        "matched_cues": cues,
        "numeric_thresholds": list(dict.fromkeys(thresholds)),
    }
