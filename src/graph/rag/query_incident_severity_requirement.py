"""Detect incident severity classification requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_INCIDENT_CONTEXT_RE = re.compile(
    r"\b(?:incidents?|outages?|service\s+degradation|breach|security\s+event|on[-\s]?call|pager|escalation)\b",
    re.I,
)
_SEVERITY_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("severity_level", (r"\bsev(?:erity)?[-\s]?[0-5]\b", r"\bseverity\s+levels?\b")),
    (
        "sev_definition",
        (
            r"\bsev[-\s]?[12]\s+(?:definitions?|criteria|meaning)\b",
            r"\bdefine\s+sev[-\s]?[12]\b",
            r"\bdefinitions?\s+for\s+sev[-\s]?[12]\b",
        ),
    ),
    (
        "severity_matrix",
        (
            r"\bseverity\s+matrix\b",
            r"\bseverity\s+classification\b",
            r"\bseverity\s+rating\b",
            r"\bincident\s+classification\b",
        ),
    ),
    (
        "priority_classification",
        (
            r"\bpriority\s+classification\b",
            r"\bpriority\s+levels?\b",
            r"\bp[0-5]\b",
            r"\bcritical\s+priority\b",
        ),
    ),
    (
        "impact_tier",
        (
            r"\bimpact\s+tiers?\b",
            r"\bcustomer\s+impact\s+tiers?\b",
            r"\bbusiness\s+impact\s+tiers?\b",
            r"\bblast\s+radius\b",
        ),
    ),
)
_ESCALATION_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "escalation_threshold",
        (
            r"\bescalation\s+thresholds?\b",
            r"\bthresholds?\s+for\s+escalat(?:e|ion)\b",
            r"\bwhen\s+to\s+escalate\b",
        ),
    ),
    (
        "response_time",
        (
            r"\bseverity[-\s]?based\s+response\s+(?:times?|targets?|windows?)\b",
            r"\bresponse\s+(?:times?|targets?|windows?)\s+by\s+severity\b",
            r"\b(?:sev|p)\s*[0-5]\b(?:\s+\w+){0,4}\s+(?:response|respond|within|in|by)\s+\d+(?:\.\d+)?\s*(?:minutes?|hours?|days?)\b",
        ),
    ),
)
_GENERAL_POSTMORTEM_RE = re.compile(r"\b(?:post[-\s]?mortem|post[-\s]?incident\s+review|lessons\s+learned|root[-\s]?cause)\b", re.I)


def detect_query_incident_severity_requirement(query: str) -> dict[str, Any]:
    """Return incident severity classification signals mentioned by a query."""
    text = _normalize_query(query)
    severity_terms = _matched_categories(_SEVERITY_SPECS, text)
    escalation_terms = _matched_categories(_ESCALATION_SPECS, text)
    matched_phrases = _unique_matches(text, _all_patterns())
    requires = _requires_incident_severity(text, severity_terms, escalation_terms)

    if not requires:
        return {
            "requires_incident_severity": False,
            "severity_terms": [],
            "escalation_terms": [],
            "matched_phrases": [],
            "recommendations": [],
            "confidence": "none",
        }

    return {
        "requires_incident_severity": True,
        "severity_terms": severity_terms,
        "escalation_terms": escalation_terms,
        "matched_phrases": matched_phrases,
        "recommendations": _recommendations(severity_terms, escalation_terms),
        "confidence": _confidence(severity_terms, escalation_terms),
    }


def _requires_incident_severity(text: str, severity_terms: list[str], escalation_terms: list[str]) -> bool:
    if not severity_terms and not escalation_terms:
        return False
    if _GENERAL_POSTMORTEM_RE.search(text) and not severity_terms and not escalation_terms:
        return False
    if "severity_level" in severity_terms:
        return True
    if escalation_terms and _INCIDENT_CONTEXT_RE.search(text):
        return True
    return bool(severity_terms) and _INCIDENT_CONTEXT_RE.search(text)


def _matched_categories(specs: tuple[tuple[str, tuple[str, ...]], ...], text: str) -> list[str]:
    return [category for category, patterns in specs if _first_match(patterns, text)]


def _unique_matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    rows = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            phrase = " ".join(match.group(0).split())
            key = phrase.casefold()
            if key not in seen:
                seen.add(key)
                rows.append((match.start(), match.end(), phrase))
    kept: list[tuple[int, int, str]] = []
    for start, end, phrase in sorted(rows, key=lambda row: (row[0], -(row[1] - row[0]))):
        if any(existing_start <= start and end <= existing_end for existing_start, existing_end, _ in kept):
            continue
        kept.append((start, end, phrase))
    return [phrase for _, _, phrase in sorted(kept, key=lambda row: row[0])]


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _all_patterns() -> tuple[str, ...]:
    return tuple(pattern for _, patterns in (*_SEVERITY_SPECS, *_ESCALATION_SPECS) for pattern in patterns)


def _recommendations(severity_terms: list[str], escalation_terms: list[str]) -> list[str]:
    recommendations = []
    if "severity_matrix" in severity_terms or len(severity_terms) > 1:
        recommendations.append("Retrieve a severity matrix mapping incident levels to impact and urgency.")
    else:
        recommendations.append("Retrieve explicit Sev1/Sev2 definitions or priority classification criteria.")
    if "impact_tier" in severity_terms:
        recommendations.append("Include evidence for impact tiers such as customer impact, business impact, or blast radius.")
    if escalation_terms:
        recommendations.append("Include severity-based response-time or escalation threshold evidence.")
    return recommendations


def _confidence(severity_terms: list[str], escalation_terms: list[str]) -> str:
    if escalation_terms and severity_terms:
        return "high"
    if "sev_definition" in severity_terms or "severity_matrix" in severity_terms:
        return "high"
    if severity_terms:
        return "medium"
    return "low"


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
