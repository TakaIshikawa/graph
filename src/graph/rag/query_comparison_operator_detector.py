"""Detect comparison operators in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SPECS: tuple[tuple[str, str, float, re.Pattern[str]], ...] = (
    ("numeric_greater_than", "numeric", 0.94, re.compile(r"(?:>=|>|at\s+least|greater\s+than|more\s+than|above|over)\b", re.I)),
    ("numeric_less_than", "numeric", 0.94, re.compile(r"(?:<=|<|at\s+most|less\s+than|fewer\s+than|below|under)\b", re.I)),
    ("numeric_equal", "numeric", 0.82, re.compile(r"\b(?:equal\s+to|equals?|exactly)\b", re.I)),
    ("temporal_before", "temporal", 0.9, re.compile(r"\b(?:before|earlier\s+than|prior\s+to|older\s+than)\b", re.I)),
    ("temporal_after", "temporal", 0.9, re.compile(r"\b(?:after|later\s+than|since|newer\s+than)\b", re.I)),
    ("ranking_top", "ranking", 0.88, re.compile(r"\b(?:top|best|highest|largest|most)\s+(?:\d+\s+)?\w*", re.I)),
    ("ranking_bottom", "ranking", 0.88, re.compile(r"\b(?:bottom|worst|lowest|smallest|least)\s+(?:\d+\s+)?\w*", re.I)),
    ("versus", "versus", 0.92, re.compile(r"\b(?:versus|vs\.?|compared\s+with|compared\s+to|against)\b", re.I)),
)


def detect_query_comparison_operators(query: str) -> list[dict[str, Any]]:
    """Return normalized comparison operator records found in a query."""
    text = " ".join(str(query or "").split())
    if not text:
        return []

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for kind, operator_class, confidence, pattern in _SPECS:
        for match in pattern.finditer(text):
            key = (match.start(), match.end(), kind)
            if key in seen:
                continue
            seen.add(key)
            matched = match.group(0).strip()
            rows.append(
                {
                    "matched_text": matched,
                    "operator_kind": kind,
                    "operator_class": operator_class,
                    "confidence": confidence,
                    "span": [match.start(), match.end()],
                }
            )
    return sorted(rows, key=lambda row: (row["span"][0], row["span"][1], row["operator_kind"]))
