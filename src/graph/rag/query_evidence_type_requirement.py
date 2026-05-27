"""Detect requested evidence types in a RAG query."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("peer_reviewed_study", re.compile(r"\b(?:peer[- ]reviewed|clinical trials?|systematic reviews?|research studies?)\b", re.I)),
    ("primary_source", re.compile(r"\b(?:primary sources?|original documents?|first[- ]hand|raw records?)\b", re.I)),
    ("example", re.compile(r"\b(?:examples?|sample cases?)\b", re.I)),
    ("benchmark", re.compile(r"\b(?:benchmarks?|performance tests?|latency tests?)\b", re.I)),
    ("official_docs", re.compile(r"\b(?:official docs?|official documentation|vendor docs?|api docs?)\b", re.I)),
    ("statistics", re.compile(r"\b(?:statistics|stats|survey data|percentages?|rates?)\b", re.I)),
    ("case_study", re.compile(r"\b(?:case studies|case study|customer stories)\b", re.I)),
)


def detect_query_evidence_type_requirements(query: object) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    requirements = []
    for label, pattern in _PATTERNS:
        spans = []
        for match in pattern.finditer(text):
            spans.append({"text": match.group(0), "start": match.start(), "end": match.end()})
        if spans:
            requirements.append({"label": label, "matched_spans": spans, "explicit": True})
    return {
        "has_evidence_type_requirement": bool(requirements),
        "requirements": requirements,
        "requirement_labels": [item["label"] for item in requirements],
    }
