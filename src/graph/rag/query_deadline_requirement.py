"""Detect deadline requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_EXACT_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.I,
)
_RELATIVE_RE = re.compile(
    r"\b(?:by|before|prior\s+to|no\s+later\s+than|due\s+(?:by|on)|deadline\s+(?:is|of|by)|within)\s+"
    r"((?:\d+\s+(?:business\s+)?(?:hours?|days?|weeks?|months?))|(?:today|tomorrow|tonight|eod|cob|friday|monday|tuesday|wednesday|thursday|saturday|sunday)|(?:launch|release|quarter\s+end|month\s+end|close))\b",
    re.I,
)
_URGENCY_RE = re.compile(r"\b(?:asap|urgent(?:ly)?|immediately|right\s+away|same\s+day|eod|cob)\b", re.I)
_HISTORICAL_CONTEXT_RE = re.compile(r"\b(?:in|during|from|since|between|after)\s+\d{4}\b", re.I)
_AMBIGUOUS_TERMS = {"today", "tomorrow", "tonight", "eod", "cob", "friday", "monday", "tuesday", "wednesday", "thursday", "saturday", "sunday", "launch", "release", "quarter end", "month end"}


def detect_query_deadline_requirement(query: str) -> dict[str, Any]:
    """Return deadline and urgency signals found in a query."""
    text = _inline_text(query)
    matched: list[dict[str, Any]] = []
    deadlines: list[dict[str, Any]] = []
    warnings: list[str] = []

    for pattern, kind in ((_URGENCY_RE, "urgency"), (_RELATIVE_RE, "relative_deadline"), (_EXACT_DATE_RE, "exact_date")):
        for match in pattern.finditer(text):
            cue = match.group(0).strip()
            if kind == "exact_date" and _looks_historical(text, match.start()):
                continue
            value = match.group(1).strip() if kind == "relative_deadline" else cue
            row = {"cue": cue, "type": kind, "span": [match.start(), match.end()]}
            if row not in matched:
                matched.append(row)
            deadlines.append({"text": value, "type": kind, "span": [match.start(), match.end()]})
            if kind == "relative_deadline" and value.casefold() in _AMBIGUOUS_TERMS:
                warnings.append(f"ambiguous_relative_deadline:{value.casefold()}")

    matched.sort(key=lambda row: (row["span"][0], row["span"][1], row["type"]))
    deadlines.sort(key=lambda row: (row["span"][0], row["span"][1], row["type"]))
    urgency = "none"
    if any(row["type"] == "urgency" for row in matched):
        urgency = "high"
    elif any(row["type"] in {"relative_deadline", "exact_date"} for row in matched):
        urgency = "medium"
    return {
        "has_deadline_requirement": bool(matched),
        "urgency_level": urgency,
        "matched_cues": matched,
        "extracted_deadlines": deadlines,
        "warnings": sorted(set(warnings)),
    }


def _looks_historical(text: str, start: int) -> bool:
    prefix = text[max(0, start - 16) : start + 4]
    return bool(_HISTORICAL_CONTEXT_RE.search(prefix))


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
