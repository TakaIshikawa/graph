"""Detect latency and SLA requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("latency", re.compile(r"\blatency\b", re.I)),
    ("p95", re.compile(r"\bp(?:50|75|90|95|99)\b", re.I)),
    ("real_time", re.compile(r"\breal[-\s]?time\b", re.I)),
    ("response_time", re.compile(r"\bresponse\s+time\b", re.I)),
    ("timeout", re.compile(r"\btimeout\b|\btimes?\s+out\b", re.I)),
    ("sla", re.compile(r"\bsla\b|service\s+level\s+agreement", re.I)),
)
_TARGET_RE = re.compile(
    r"\b(?:under|below|less\s+than|within|<=?|p(?:50|75|90|95|99)\s*(?:latency)?\s*(?:under|below|less\s+than|within|<=?)?)\s*\d+(?:\.\d+)?\s*(?:ms|milliseconds?|s|sec(?:ond)?s?)\b",
    re.I,
)


def detect_query_latency_sla_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    signals = [name for name, pattern in _SIGNALS if pattern.search(text)]
    targets = _latency_targets(text)
    return {"requires_latency_sla": bool(signals or targets), "latency_targets": targets, "signals": signals}


def _latency_targets(text: str) -> list[str]:
    seen: set[str] = set()
    targets: list[str] = []
    for match in _TARGET_RE.finditer(text):
        value = match.group(0).strip()
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            targets.append(value)
    return targets
