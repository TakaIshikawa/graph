"""Detect latency requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_NUMERIC_LIMIT_RE = re.compile(
    r"\b(?:within|under|below|less\s+than|in)\s+(\d+(?:\.\d+)?)\s*"
    r"(ms|milliseconds?|s|sec(?:ond)?s?|m|min(?:ute)?s?|h|hours?|hrs?)\b",
    re.I,
)
_CUE_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("real_time", "real_time", re.compile(r"\b(?:real[-\s]?time|instant|immediate|low[-\s]?latency)\b", re.I)),
    ("fast", "fast", re.compile(r"\b(?:fast|quick|rapid|snappy)\b", re.I)),
    ("overnight", "overnight", re.compile(r"\b(?:overnight|by\s+tomorrow)\b", re.I)),
    ("async", "async", re.compile(r"\b(?:async|asynchronous|background\s+job)\b", re.I)),
    ("batch", "batch", re.compile(r"\b(?:batch|bulk|scheduled\s+job)\b", re.I)),
    ("can_wait", "can_wait", re.compile(r"\b(?:can\s+wait|not\s+urgent|no\s+rush)\b", re.I)),
)
_CLASS_PRIORITY = ("real_time", "fast", "batch", "async", "overnight", "can_wait")
_UNIT_SECONDS = {
    "ms": 0.001,
    "millisecond": 0.001,
    "milliseconds": 0.001,
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
}


def detect_query_latency_requirement(query: str) -> dict[str, Any]:
    """Return latency class, numeric limits, and cue spans."""
    text = _inline_text(query)
    cues = _cues(text)
    numeric_limits = _numeric_limits(text)
    classes = {cue["latency_class"] for cue in cues}
    if numeric_limits and not classes:
        classes.add("fast")
    latency_class = next((name for name in _CLASS_PRIORITY if name in classes), "unconstrained")
    return {
        "requires_latency_awareness": bool(cues or numeric_limits),
        "latency_class": latency_class,
        "numeric_limits": numeric_limits,
        "cues": cues,
    }


def _cues(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue_type, latency_class, pattern in _CUE_SPECS:
        for match in pattern.finditer(text):
            rows.append({"cue": match.group(0), "type": cue_type, "latency_class": latency_class, "span": [match.start(), match.end()]})
    return sorted(rows, key=lambda row: (row["span"][0], row["span"][1], row["type"]))


def _numeric_limits(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in _NUMERIC_LIMIT_RE.finditer(text):
        value = float(match.group(1))
        unit = match.group(2).lower()
        seconds = value * _UNIT_SECONDS[unit]
        rows.append(
            {
                "cue": match.group(0),
                "value": int(value) if value.is_integer() else value,
                "unit": unit,
                "seconds": seconds,
                "span": [match.start(), match.end()],
            }
        )
    return rows


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
