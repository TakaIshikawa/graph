"""Detect requested evidence methodologies in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("randomized_trial", re.compile(r"\b(?:randomi[sz]ed controlled trials?|rcts?|clinical trials?)\b", re.I)),
    ("cohort_study", re.compile(r"\b(?:cohort studies|cohort study|longitudinal studies|prospective cohorts?)\b", re.I)),
    ("survey", re.compile(r"\b(?:surveys?|polls?|questionnaires?)\b", re.I)),
    ("benchmark", re.compile(r"\b(?:benchmarks?|benchmarking|performance tests?|load tests?)\b", re.I)),
    ("case_study", re.compile(r"\b(?:case studies|case study|case reports?)\b", re.I)),
    ("audit", re.compile(r"\b(?:independent audits?|security audits?|compliance audits?|audits)\b", re.I)),
    ("meta_analysis", re.compile(r"\b(?:meta[- ]analyses|meta[- ]analysis|systematic reviews?)\b", re.I)),
)


def detect_query_methodology_requirements(query: str) -> dict[str, Any]:
    """Return requested source methodology categories and matched spans."""
    text = " ".join(("" if query is None else str(query)).split())
    requirements = []
    for category, pattern in _PATTERNS:
        spans = [{"text": m.group(0), "start": m.start(), "end": m.end()} for m in pattern.finditer(text)]
        if spans:
            requirements.append({"category": category, "matched_spans": spans})
    return {
        "has_methodology_requirement": bool(requirements),
        "methodology_categories": [item["category"] for item in requirements],
        "requirements": requirements,
    }
