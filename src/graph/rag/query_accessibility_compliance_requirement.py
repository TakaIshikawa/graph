"""Detect accessibility compliance requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_STANDARDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("WCAG", re.compile(r"\bwcag(?:\s*2(?:\.\d)?)?\b|web\s+content\s+accessibility\s+guidelines", re.I)),
    ("ADA", re.compile(r"\bada\b|americans\s+with\s+disabilities\s+act", re.I)),
    ("Section 508", re.compile(r"\bsection\s+508\b", re.I)),
)
_TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("wcag", _STANDARDS[0][1]),
    ("ada", _STANDARDS[1][1]),
    ("screen_reader", re.compile(r"\b(?:screen\s+reader|voiceover|nvda|jaws)\b", re.I)),
    ("keyboard_navigation", re.compile(r"\b(?:keyboard\s+navigation|keyboard\s+accessible|tab\s+order|focus\s+(?:state|management))\b", re.I)),
    ("captions", re.compile(r"\b(?:captions?|closed\s+captions?|subtitles?)\b", re.I)),
    ("alt_text", re.compile(r"\b(?:alt\s+text|alternative\s+text|image\s+description)\b", re.I)),
    ("color_contrast", re.compile(r"\b(?:color\s+contrast|colour\s+contrast|contrast\s+ratio)\b", re.I)),
    ("reduced_motion", re.compile(r"\b(?:reduced\s+motion|motion\s+sensitivity|avoid\s+animations?)\b", re.I)),
    ("accessible_document", re.compile(r"\b(?:accessible\s+(?:pdf|document)|tagged\s+pdf|pdf/ua)\b", re.I)),
)


def detect_query_accessibility_compliance_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    matches = _matches(text)
    terms = _ordered_unique(match["term"] for match in matches)
    standards = _detected_standards(text)
    requires = bool(terms or standards)
    return {
        "requires_accessibility_compliance": requires,
        "accessibility_terms": terms,
        "matched_phrases": [match["phrase"] for match in matches],
        "standards": standards,
        "recommendations": _recommendations(standards, terms),
        "confidence": _confidence(standards, terms),
    }


def _matches(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for term, pattern in _TERMS:
        for match in pattern.finditer(text):
            phrase = match.group(0)
            key = (term, phrase.casefold())
            if key not in seen:
                rows.append({"term": term, "phrase": phrase, "span": [match.start(), match.end()]})
                seen.add(key)
    return sorted(rows, key=lambda row: (row["span"][0], row["span"][1], row["term"]))


def _detected_standards(text: str) -> list[str]:
    return [name for name, pattern in _STANDARDS if pattern.search(text)]


def _recommendations(standards: list[str], terms: list[str]) -> list[str]:
    recommendations: list[str] = []
    if standards:
        recommendations.append("retrieve_named_accessibility_standards_documents")
    implementation_terms = [term for term in terms if term not in {"wcag", "ada"}]
    if implementation_terms:
        recommendations.append("retrieve_implementation_evidence_for_accessibility_features")
    if "accessible_document" in terms:
        recommendations.append("include_accessible_pdf_or_document_evidence")
    return recommendations


def _confidence(standards: list[str], terms: list[str]) -> float:
    if standards and len(terms) >= 2:
        return 0.95
    if standards:
        return 0.85
    if len(terms) >= 2:
        return 0.8
    if terms:
        return 0.65
    return 0.0


def _ordered_unique(values: list[str] | Any) -> list[str]:
    rows: list[str] = []
    for value in values:
        if value not in rows:
            rows.append(value)
    return rows
