"""Detect locale-sensitive interpretation needs in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_LOCALE_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("language_variant", re.compile(r"\b(?:british|american|canadian|australian) english\b|\blanguage variant\b", re.I)),
    ("country_specific", re.compile(r"\b(?:country-specific|for (?:the )?(?:us|u\.s\.|uk|u\.k\.|canada|australia|germany|france|japan))\b", re.I)),
    ("locale", re.compile(r"\blocale\b|\blocal format\b|\bregional format\b", re.I)),
)
_FORMAT_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("date_format", re.compile(r"\bdd/mm/yyyy\b|\bmm/dd/yyyy\b|\byyyy-mm-dd\b|\bdate format\b", re.I)),
    ("address_format", re.compile(r"\baddress format\b|\bstreet address\b|\bpostcode before city\b", re.I)),
    ("postal_code", re.compile(r"\bpostal code\b|\bpostcode\b|\bzip code\b", re.I)),
    ("decimal_separator", re.compile(r"\bdecimal separator\b|\bcomma decimal\b|\bdecimal comma\b|\b1,23\b", re.I)),
    ("measurement_convention", re.compile(r"\bmetric\b|\bimperial\b|\bmiles?\b|\bkilometers?\b|\bcelsius\b|\bfahrenheit\b", re.I)),
)


def detect_query_locale_requirement(query: str) -> dict[str, Any]:
    """Return locale and formatting cues with normalization recommendations."""
    normalized = _normalize_query(query)
    locale_cues = [label for label, pattern in _LOCALE_CUES if pattern.search(normalized)]
    format_cues = [label for label, pattern in _FORMAT_CUES if pattern.search(normalized)]
    requires = bool(locale_cues or format_cues)
    recommendations = []
    if locale_cues:
        recommendations.append("resolve_requested_locale_before_retrieval_and_formatting")
    if "date_format" in format_cues:
        recommendations.append("normalize_ambiguous_dates_to_iso_8601")
    if any(cue in format_cues for cue in ("address_format", "postal_code")):
        recommendations.append("use_country_specific_address_and_postal_code_parsing")
    if any(cue in format_cues for cue in ("decimal_separator", "measurement_convention")):
        recommendations.append("preserve_locale_units_and_number_separators_in_answer")
    return {
        "requires_locale_awareness": requires,
        "locale_cues": locale_cues,
        "format_cues": format_cues,
        "normalization_recommendations": recommendations,
        "confidence": 0.85 if locale_cues else (0.65 if format_cues else 0.0),
        "normalized_query": normalized,
    }


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.casefold().split())
