"""Detect model evaluation requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_EVALUATION_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("benchmark_suite", (r"\bbenchmark\s+suite\b", r"\bevaluation\s+benchmark\b")),
    ("evals", (r"\bevals\b", r"\bmodel\s+evaluations?\b", r"\brag\s+evals?\b", r"\brag\s+evaluations?\b")),
    ("golden_dataset", (r"\bgolden\s+datasets?\b", r"\bgold\s+standard\s+datasets?\b")),
    ("quality_gate", (r"\bquality\s+gates?\b", r"\bgo/no-go\s+criteria\b")),
    ("regression_eval", (r"\bregression\s+evals?\b", r"\bregression\s+evaluations?\b")),
)
_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("latency", (r"\blatency\b",)),
    ("pass_rate", (r"\bpass\s+rate\b", r"\bpass-rate\b")),
    ("precision", (r"\bprecision\b",)),
    ("recall", (r"\brecall\b",)),
)


def detect_query_model_evaluation_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    evaluation_types = []
    matched_cues = []
    for category, patterns in _EVALUATION_TYPES:
        match = _first_match(patterns, text)
        if match:
            evaluation_types.append(category)
            matched_cues.append({"category": category, "matched_text": match.group(0)})
    metrics = [name for name, patterns in _METRICS if _matches_any(patterns, text)]

    requires_evaluation = bool(evaluation_types)
    return {
        "requires_model_evaluation": requires_evaluation,
        "evaluation_types": evaluation_types,
        "metrics": metrics,
        "matched_cues": matched_cues,
        "confidence": "high" if len(evaluation_types) > 1 else ("medium" if requires_evaluation else "none"),
    }


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
