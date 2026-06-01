"""Detect citation and source requirements in user queries."""

from __future__ import annotations

import re
from typing import Any

_NO_CITATION = re.compile(r"\b(no|without|don't|do not)\s+(?:citations?|sources?|links?)\b", re.I)
_TERMS = (
    ("cite sources", "citation", re.compile(r"\bcite sources?\b|\bcitations?\b|\bsources?\b", re.I)),
    ("include links", "citation", re.compile(r"\binclude links?\b|\blinks to\b", re.I)),
    ("quote evidence", "quote", re.compile(r"\bquote evidence\b|\bquoted?\b", re.I)),
    ("primary sources", "primary", re.compile(r"\bprimary sources?\b|\bofficial sources?\b", re.I)),
    ("peer-reviewed", "primary", re.compile(r"\bpeer[-\s]?reviewed\b", re.I)),
    ("bibliography", "citation", re.compile(r"\bbibliograph(?:y|ies)\b|\breferences\b", re.I)),
)


def detect_query_citation_requirement(query: str) -> dict[str, Any]:
    text = str(query or "")
    excludes = bool(_NO_CITATION.search(text))
    matched = [label for label, _kind, pattern in _TERMS if pattern.search(text)]
    kinds = {kind for _label, kind, pattern in _TERMS if pattern.search(text)}
    confidence = 0.0 if not matched and not excludes else min(1.0, 0.35 + 0.2 * len(matched) + (0.25 if excludes else 0))
    return {
        "requires_citations": bool(matched) and not excludes,
        "requires_quotes": "quote" in kinds and not excludes,
        "requires_primary_sources": "primary" in kinds and not excludes,
        "excludes_citations": excludes,
        "matched_terms": matched,
        "confidence": round(confidence, 2),
    }
