"""Detect conflict-of-interest and independence requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(r"\b(?:sources?|evidence|stud(?:y|ies)|papers?|authors?|research|citations?|results?)\b", re.I)
_SPECS: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    ("funding", re.compile(r"\b(?:funding sources?|funded by|grant support|financial backing)\b", re.I), False),
    ("sponsor", re.compile(r"\b(?:sponsors?|sponsored by|industry sponsored|vendor sponsored)\b", re.I), False),
    ("affiliation", re.compile(r"\b(?:author affiliations?|institutional affiliations?|company affiliations?)\b", re.I), False),
    ("disclosure", re.compile(r"\b(?:disclosures?|disclosed interests?|competing interests?)\b", re.I), False),
    ("conflict_of_interest", re.compile(r"\b(?:conflicts? of interest|coi)\b", re.I), False),
    ("independence", re.compile(r"\b(?:independent|independently funded|third[- ]party)\b", re.I), True),
)


def detect_query_conflict_of_interest_requirements(query: str) -> dict[str, Any]:
    text = " ".join(("" if query is None else str(query)).split())
    has_context = bool(_CONTEXT_RE.search(text))
    requirements = []
    for category, pattern, needs_context in _SPECS:
        if needs_context and not has_context:
            continue
        matches = [{"text": m.group(0), "start": m.start(), "end": m.end()} for m in pattern.finditer(text)]
        if matches:
            requirements.append({"category": category, "matched_spans": matches})
    return {
        "requires_conflict_of_interest_check": bool(requirements),
        "categories": [item["category"] for item in requirements],
        "requirements": requirements,
        "matched_phrases": [span["text"] for item in requirements for span in item["matched_spans"]],
    }
