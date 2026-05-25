"""Detect geospatial precision requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PRECISION_LEVELS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("country", re.compile(r"\bcountry-level\b|\bby country\b", re.I)),
    ("state_region", re.compile(r"\bstate-level\b|\bprovince\b|\bregion\b", re.I)),
    ("county", re.compile(r"\bcounty-level\b|\bcounty\b", re.I)),
    ("city", re.compile(r"\bcity-level\b|\bcity\b|\bmunicipal\b", re.I)),
    ("neighborhood", re.compile(r"\bneighbou?rhood\b|\bdistrict\b", re.I)),
    ("postal_code", re.compile(r"\bzip code\b|\bpostal code\b|\bpostcode\b", re.I)),
    ("address", re.compile(r"\bexact address\b|\bstreet address\b|\baddress\b", re.I)),
    ("coordinates", re.compile(r"\bcoordinates?\b|\blatitude\b|\blongitude\b|\blat\b|\blon\b", re.I)),
    ("bounding_box", re.compile(r"\bbounding box\b|\bbbox\b", re.I)),
)
_DISTANCE_RE = re.compile(r"\bwithin\s+(\d+(?:\.\d+)?)\s*(miles?|mi|kilometers?|km|meters?|m)\b", re.I)
_COORD_RE = re.compile(r"\b(?:lat(?:itude)?|lon(?:gitude)?|coordinates?)\b", re.I)


def detect_query_geospatial_precision_requirement(query: str) -> dict[str, Any]:
    """Return geospatial precision cues, distances, and retrieval recommendations."""
    normalized = _normalize_query(query)
    levels = [label for label, pattern in _PRECISION_LEVELS if pattern.search(normalized)]
    distances = [
        {"value": match.group(1), "unit": _normalize_unit(match.group(2))}
        for match in _DISTANCE_RE.finditer(normalized)
    ]
    coordinate_cues = sorted({match.group(0).casefold() for match in _COORD_RE.finditer(normalized)})
    requires = bool(levels or distances or coordinate_cues)
    recommendations = []
    if levels:
        recommendations.append("retrieve_sources_with_matching_location_granularity")
    if distances:
        recommendations.append("apply_radius_filter_before_ranking_results")
    if coordinate_cues:
        recommendations.append("preserve_coordinate_precision_and_coordinate_reference_system")
    return {
        "requires_geospatial_precision": requires,
        "precision_levels": levels,
        "distance_constraints": distances,
        "coordinate_cues": coordinate_cues,
        "recommendations": recommendations,
        "confidence": 0.85 if coordinate_cues or distances else (0.65 if levels else 0.0),
        "normalized_query": normalized,
    }


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.casefold().split())


def _normalize_unit(unit: str) -> str:
    folded = unit.casefold()
    if folded in {"mi", "mile", "miles"}:
        return "miles"
    if folded in {"km", "kilometer", "kilometers"}:
        return "kilometers"
    return "meters"
