"""Classify evidence requirements implied by a RAG query."""

from __future__ import annotations

import re
from typing import Any

_FLAGS = (
    "requires_recency",
    "requires_comparison",
    "requires_source_diversity",
    "requires_citations",
    "requires_quantitative_evidence",
    "requires_synthesis_steps",
)

_CUES: dict[str, tuple[tuple[str, str], ...]] = {
    "requires_recency": (
        ("latest", r"\blatest\b"),
        ("current", r"\bcurrent(?:ly)?\b"),
        ("recent", r"\brecent(?:ly)?\b"),
        ("today", r"\btoday\b"),
        ("this-year", r"\bthis\s+year\b"),
        ("up-to-date", r"\bup[- ]to[- ]date\b"),
    ),
    "requires_comparison": (
        ("compare", r"\bcompar(?:e|ed|es|ing|ison)\b"),
        ("versus", r"\b(?:vs\.?|versus)\b"),
        ("difference", r"\bdifferen(?:ce|ces|t)\b"),
        ("better", r"\bbetter\b"),
        ("tradeoff", r"\btrade[- ]offs?\b"),
    ),
    "requires_source_diversity": (
        ("multiple-sources", r"\bmultiple\s+sources\b"),
        ("different-sources", r"\bdifferent\s+sources\b"),
        ("cross-check", r"\bcross[- ]check\b"),
        ("consensus", r"\bconsensus\b"),
        ("conflicting", r"\bconflicting\b"),
    ),
    "requires_citations": (
        ("cite", r"\bcit(?:e|es|ation|ations)\b"),
        ("sources", r"\bsources?\b"),
        ("references", r"\breferences?\b"),
        ("footnotes", r"\bfootnotes?\b"),
        ("links", r"\blinks?\b"),
    ),
    "requires_quantitative_evidence": (
        ("how-many", r"\bhow\s+many\b"),
        ("how-much", r"\bhow\s+much\b"),
        ("percentage", r"\bpercent(?:age)?s?\b"),
        ("metrics", r"\bmetrics?\b"),
        ("statistics", r"\bstat(?:s|istics)\b"),
        ("numbers", r"\bnumbers?\b"),
        ("rate", r"\brates?\b"),
    ),
    "requires_synthesis_steps": (
        ("step-by-step", r"\bstep[- ]by[- ]step\b"),
        ("plan", r"\bplans?\b"),
        ("strategy", r"\bstrateg(?:y|ies)\b"),
        ("synthesize", r"\bsynthesi(?:s|ze)\b"),
        ("pros-cons", r"\bpros\s+and\s+cons\b"),
        ("recommendation", r"\brecommend(?:ation|ations|ed)?\b"),
    ),
}


def classify_query_evidence_requirements(query: str) -> dict[str, Any]:
    """Return deterministic evidence requirement flags and cue reasons for a query."""
    normalized = _normalize_query(query)
    reasons = {
        flag: [
            f"cue:{label}"
            for label, pattern in _CUES[flag]
            if re.search(pattern, normalized)
        ]
        for flag in _FLAGS
    }

    payload: dict[str, Any] = {
        flag: bool(reasons[flag])
        for flag in _FLAGS
    }
    payload["reasons"] = reasons
    payload["normalized_query"] = normalized
    return payload


def _normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a non-empty string")
    normalized = " ".join(query.casefold().strip().split())
    if not normalized:
        raise ValueError("query must be a non-empty string")
    return normalized
