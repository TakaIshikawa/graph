"""Detect license and reuse requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_LICENSES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("open_license", re.compile(r"\bopen license\b|\bopenly licensed\b", re.I)),
    ("creative_commons", re.compile(r"\bcreative commons\b|\bcc[- ]?by\b|\bcc0\b", re.I)),
    ("public_domain", re.compile(r"\bpublic domain\b", re.I)),
    ("fair_use", re.compile(r"\bfair use\b", re.I)),
    ("proprietary", re.compile(r"\bproprietary\b", re.I)),
)
_REUSE: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("commercial_use", re.compile(r"\bcommercial use\b|\bfor commercial\b", re.I)),
    ("redistribution", re.compile(r"\bredistribut(?:e|ion)\b|\brepublish\b", re.I)),
    ("attribution_required", re.compile(r"\battribution required\b|\bwith attribution\b", re.I)),
)
_RESTRICTED: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("proprietary", re.compile(r"\bproprietary\b", re.I)),
    ("noncommercial_only", re.compile(r"\bnon[- ]commercial\b|\bnot for commercial use\b", re.I)),
    ("fair_use_only", re.compile(r"\bfair use\b", re.I)),
)


def detect_query_source_licensing_requirement(query: str) -> dict[str, Any]:
    """Return source licensing and reuse-filtering requirements."""
    normalized = _normalize_query(query)
    licenses = [label for label, pattern in _LICENSES if pattern.search(normalized)]
    reuse = [label for label, pattern in _REUSE if pattern.search(normalized)]
    restricted = [label for label, pattern in _RESTRICTED if pattern.search(normalized)]
    requires = bool(licenses or reuse or restricted)
    recommendations = []
    if requires:
        recommendations.append("filter_sources_by_machine_readable_license_metadata")
    if reuse:
        recommendations.append("preserve_attribution_and_reuse_terms_in_citations")
    if restricted:
        recommendations.append("exclude_or_flag_sources_with_restricted_reuse_terms")
    return {
        "requires_license_filtering": requires,
        "license_cues": licenses,
        "reuse_cues": reuse,
        "restricted_use_cues": restricted,
        "recommendations": recommendations,
        "confidence": 0.85 if licenses or restricted else (0.65 if reuse else 0.0),
        "normalized_query": normalized,
    }


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.casefold().split())
