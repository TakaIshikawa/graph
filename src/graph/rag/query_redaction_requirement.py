"""Detect whether a RAG query asks for redaction or sensitive entity removal."""

from __future__ import annotations

import re
from typing import Any

_REDACTION_TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anonymization", re.compile(r"\banonymi[sz](?:e|ation|ed|ing)\b", re.I)),
    ("de-identification", re.compile(r"\bde[-\s]?identif(?:y|ication|ied|ying)\b", re.I)),
    ("masking", re.compile(r"\bmask(?:ing|ed)?\b", re.I)),
    ("pii_removal", re.compile(r"\b(?:remove|strip|delete)\s+(?:pii|personal(?:ly identifiable)? information)\b", re.I)),
    ("redaction", re.compile(r"\bredact(?:ion|ed|ing)?\b", re.I)),
    ("secret_removal", re.compile(r"\b(?:remove|strip|delete)\s+(?:secrets?|tokens?|api keys?|credentials?)\b", re.I)),
)
_SENSITIVE_ENTITY_TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("api_keys", re.compile(r"\bapi keys?\b", re.I)),
    ("credentials", re.compile(r"\bcredentials?\b", re.I)),
    ("emails", re.compile(r"\bemail addresses?\b|\bemails?\b", re.I)),
    ("names", re.compile(r"\b(?:full|legal|personal)?\s*names?\b", re.I)),
    ("phone_numbers", re.compile(r"\bphone numbers?\b|\btelephone numbers?\b", re.I)),
    ("pii", re.compile(r"\bpii\b|personally identifiable information|personal data", re.I)),
    ("secrets", re.compile(r"\bsecrets?\b", re.I)),
    ("ssn", re.compile(r"\bssn\b|social security", re.I)),
    ("tokens", re.compile(r"\btokens?\b", re.I)),
)


def detect_query_redaction_requirement(query: str) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    text = " ".join(query.split())
    redaction_terms = sorted({term for term, pattern in _REDACTION_TERMS if pattern.search(text)})
    sensitive_entity_terms = sorted({term for term, pattern in _SENSITIVE_ENTITY_TERMS if pattern.search(text)})
    requires_redaction = bool(redaction_terms or sensitive_entity_terms)
    rationale = _rationale(redaction_terms, sensitive_entity_terms)

    return {
        "requires_redaction": requires_redaction,
        "redaction_terms": redaction_terms,
        "sensitive_entity_terms": sensitive_entity_terms,
        "rationale": rationale,
    }


def _rationale(redaction_terms: list[str], sensitive_entity_terms: list[str]) -> str:
    if redaction_terms and sensitive_entity_terms:
        return "Query includes redaction instructions and sensitive entity terms."
    if redaction_terms:
        return "Query includes explicit redaction instructions."
    if sensitive_entity_terms:
        return "Query names sensitive entities that may require redaction."
    return "No redaction requirement detected."
