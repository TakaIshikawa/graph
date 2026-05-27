"""Detect exclusion criteria in RAG queries."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_CUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("do not include", re.compile(r"\bdo\s+not\s+include\b", re.I)),
    ("not from", re.compile(r"\bnot\s+from\b", re.I)),
    ("without", re.compile(r"\bwithout\b", re.I)),
    ("exclude", re.compile(r"\bexclud(?:e|ing)\b", re.I)),
    ("except", re.compile(r"\bexcept\b", re.I)),
    ("ignore", re.compile(r"\bignore\b", re.I)),
)
_SPAN_END_RE = re.compile(r"\s*(?:[,;.]|\band\b|\bbut\b|\bwhile\b|\bwhen\b|\bthat\b|\bwith\b)\s*", re.I)


def detect_query_exclusion_criteria(query: str) -> dict[str, Any]:
    """Return exclusion cues and query text with exclusion spans removed."""
    text = _inline_text(query)
    exclusions = _exclusions(text)
    cue_counts = dict(Counter(row["cue"] for row in exclusions))
    return {
        "has_exclusions": bool(exclusions),
        "exclusions": exclusions,
        "cue_counts": cue_counts,
        "normalized_query_without_exclusions": _without_spans(text, [row["span"] for row in exclusions]),
    }


def _exclusions(text: str) -> list[dict[str, Any]]:
    matches: list[tuple[int, int, str]] = []
    for cue, pattern in _CUE_PATTERNS:
        matches.extend((match.start(), match.end(), cue) for match in pattern.finditer(text))
    matches.sort(key=lambda row: (row[0], -(row[1] - row[0])))

    rows: list[dict[str, Any]] = []
    last_end = -1
    for start, cue_end, cue in matches:
        if start < last_end:
            continue
        end = _span_end(text, cue_end)
        raw = text[cue_end:end].strip(" \t,;:.")
        rows.append({"cue": cue, "text": raw, "span": [start, end]})
        last_end = end
    return rows


def _span_end(text: str, start: int) -> int:
    match = _SPAN_END_RE.search(text, start)
    return match.start() if match else len(text)


def _without_spans(text: str, spans: list[list[int]]) -> str:
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    normalized = " ".join("".join(pieces).split())
    return re.sub(r"\s+([,;:.])", r"\1", normalized).strip(" ,;:")


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
