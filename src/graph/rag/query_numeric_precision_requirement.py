"""Detect RAG queries that require exact numeric evidence."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_INTENTS = {
    "pricing": re.compile(r"(?i)\b(price|cost|fee|pricing|quote|budget|\$|usd|eur|gbp)\b"),
    "dosage": re.compile(r"(?i)\b(dose|dosage|mg|mcg|ml|tablet|medication)\b"),
    "benchmark": re.compile(r"(?i)\b(benchmark|latency|throughput|fps|score|performance)\b"),
    "ranking": re.compile(r"(?i)\b(rank|ranking|top\s+\d+|best|largest|smallest|highest|lowest)\b"),
    "range": re.compile(r"(?i)\b(range|between|from\s+\d|to\s+\d|minimum|maximum|min|max)\b"),
}
_PRECISION_RE = re.compile(r"(?i)\b(exact|precise|specific|nearest|decimal|significant figures?|tolerance|within|±|\+/-)\b")
_APPROX_RE = re.compile(r"(?i)\b(roughly|approximately|about|around|ballpark|estimate|high level)\b")
_UNIT_RE = re.compile(r"(?i)(\$|€|£|%|\b(?:usd|eur|gbp|mg|mcg|kg|g|ml|l|ms|s|fps|gb|mb|mph|km/h|kwh)\b)")
_NUMBER_RE = re.compile(r"\d")


def detect_query_numeric_precision_requirements(query: str) -> dict[str, Any]:
    normalized = " ".join((string(query) or "").casefold().split())
    intents = [name for name, pattern in _INTENTS.items() if pattern.search(normalized)]
    precision_words = sorted({match.group(0).casefold() for match in _PRECISION_RE.finditer(normalized)})
    unit_hints = sorted({match.group(0).casefold() for match in _UNIT_RE.finditer(normalized)})
    approximate = bool(_APPROX_RE.search(normalized))
    has_numeric_signal = bool(_NUMBER_RE.search(normalized) or intents or unit_hints)
    requires_exact = bool(precision_words or unit_hints or any(intent in intents for intent in ("pricing", "dosage", "benchmark", "ranking")) or re.search(r"\bhow many|how much\b", normalized))
    approximate_answers_risky = requires_exact and not approximate

    return {
        "numeric_intents": intents,
        "requested_precision_words": precision_words,
        "unit_hints": unit_hints,
        "requires_exact_numeric_evidence": requires_exact,
        "approximate_answer_risky": approximate_answers_risky,
        "approximate_exploratory": approximate and has_numeric_signal and not precision_words,
    }
