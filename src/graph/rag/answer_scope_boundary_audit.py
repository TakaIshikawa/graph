"""Audit answer sentences for scope boundary drift."""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&]*(?:\s+[A-Z][A-Za-z0-9&]*){0,3}\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}s?\b")
_JURISDICTIONS = ("US", "United States", "EU", "European Union", "UK", "United Kingdom", "California", "Japan", "Canada", "Germany", "France")
_GENERIC_ENTITIES = {"The", "This", "That", "It", "They", "For", "In", "A", "An"}


def audit_answer_scope_boundaries(answer: object, query: object | None = None) -> dict[str, Any]:
    query_text = str(query or "")
    answer_text = str(answer or "")
    query_entities = _entities(query_text)
    query_years = set(_YEAR_RE.findall(query_text))
    query_jurisdictions = _jurisdictions(query_text)
    issues = []

    for sentence in _sentences(answer_text):
        for entity in _entities(sentence) - query_entities - query_jurisdictions:
            issues.append(_issue(sentence, "entity_drift", f"Introduces entity '{entity}' absent from query.", "medium"))
        for year in set(_YEAR_RE.findall(sentence)) - query_years:
            issues.append(_issue(sentence, "temporal_drift", f"Introduces time range '{year}' absent from query.", "medium"))
        for jurisdiction in _jurisdictions(sentence) - query_jurisdictions:
            issues.append(_issue(sentence, "jurisdiction_drift", f"Introduces jurisdiction '{jurisdiction}' absent from query.", "high"))

    deduped = []
    seen = set()
    for issue in issues:
        key = (issue["sentence"], issue["reason"])
        if key not in seen:
            seen.add(key)
            deduped.append(issue)

    return {
        "issue_count": len(deduped),
        "severity": _severity(deduped),
        "issues": deduped,
        "summary": {
            "entity_drift": sum(1 for issue in deduped if issue["reason"] == "entity_drift"),
            "temporal_drift": sum(1 for issue in deduped if issue["reason"] == "temporal_drift"),
            "jurisdiction_drift": sum(1 for issue in deduped if issue["reason"] == "jurisdiction_drift"),
        },
    }


def _sentences(text: str) -> list[str]:
    return [" ".join(match.group(0).strip().split()) for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _entities(text: str) -> set[str]:
    return {match.group(0) for match in _ENTITY_RE.finditer(text) if match.group(0) not in _GENERIC_ENTITIES and match.group(0) not in _JURISDICTIONS}


def _jurisdictions(text: str) -> set[str]:
    folded = text.casefold()
    return {term for term in _JURISDICTIONS if re.search(rf"\b{re.escape(term.casefold())}\b", folded)}


def _issue(sentence: str, reason: str, detail: str, severity: str) -> dict[str, Any]:
    return {"severity": severity, "sentence": sentence, "reason": reason, "detail": detail}


def _severity(issues: list[dict[str, Any]]) -> str:
    if any(issue["severity"] == "high" for issue in issues):
        return "high"
    if issues:
        return "medium"
    return "none"
