"""Detect migration planning requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUE_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("migration_plan", re.compile(r"\b(?:migration\s+plan|implementation\s+migration\s+plan)\b", re.I)),
    ("cutover", re.compile(r"\bcutover\b", re.I)),
    ("rollback_plan", re.compile(r"\b(?:rollback\s+plan|roll\s+back\s+plan)\b", re.I)),
    ("phased_migration", re.compile(r"\b(?:phased\s+migration|migration\s+in\s+phases)\b", re.I)),
    ("data_migration", re.compile(r"\bdata\s+migration\b", re.I)),
    ("parallel_run", re.compile(r"\b(?:parallel\s+run|run\s+in\s+parallel)\b", re.I)),
    ("backout", re.compile(r"\bbackout\b", re.I)),
    ("go_live_checklist", re.compile(r"\b(?:go[-\s]?live\s+checklist|launch\s+checklist)\b", re.I)),
)

_TERM_PATTERN = re.compile(r"\b(?:pilot|phase\s+\d+|wave\s+\d+|wave|weekend\s+cutover|rollback\s+window)\b", re.I)


def detect_query_migration_plan_requirement(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    cue_matches = _cue_matches(normalized)
    phase_terms = _phase_terms(normalized)
    return {
        "requires_migration_plan": bool(cue_matches),
        "cue_categories": [match["category"] for match in cue_matches],
        "matched_cues": cue_matches,
        "phase_terms": phase_terms,
        "normalized_query": normalized,
    }


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.split())


def _cue_matches(normalized_query: str) -> list[dict[str, Any]]:
    rows = []
    for category, pattern in _CUE_SPECS:
        match = pattern.search(normalized_query)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["category"]))
    return rows


def _phase_terms(normalized_query: str) -> list[str]:
    seen: set[str] = set()
    terms = []
    for match in _TERM_PATTERN.finditer(normalized_query):
        term = match.group(0)
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            terms.append(term)
    return terms
