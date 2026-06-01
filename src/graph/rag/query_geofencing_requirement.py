"""Detect geofencing requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_ALLOW_PATTERNS = (r"\ballow\s+only\s+(?P<region>[a-z][a-z\s-]+?)(?:\s+users?)?\b", r"\b(?:restrict|limit)\s+access\s+to\s+(?P<region>[a-z][a-z\s-]+?)\b")
_BLOCK_PATTERNS = (r"\bblock\s+(?:users\s+)?(?:by|from)\s+(?P<region>[a-z][a-z\s-]+?)\b", r"\bdeny\s+access\s+(?:from|in)\s+(?P<region>[a-z][a-z\s-]+?)\b")
_GENERAL_CUES = (r"\bgeo[-\s]?fenc(?:e|ing)\b", r"\bip\s+geo(?:graphy|location)\b", r"\bregional\s+access\s+rules?\b")
_REGIONS = ("eu", "europe", "us", "united states", "uk", "japan", "canada", "australia", "apac", "emea", "latam", "country")


def detect_query_geofencing_requirement(query: str) -> dict[str, Any]:
    """Return geofencing signals mentioned by a query."""
    text = " ".join(str(query or "").split())
    lowered = text.casefold()
    matched_cues: list[str] = []
    regions: list[str] = []
    restriction_type = "unspecified"

    if _collect_matches(_ALLOW_PATTERNS, text, matched_cues, regions):
        restriction_type = "allowlist"
    if _collect_matches(_BLOCK_PATTERNS, text, matched_cues, regions):
        restriction_type = "mixed" if restriction_type == "allowlist" else "blocklist"
    if any(re.search(pattern, text, re.I) for pattern in _GENERAL_CUES):
        matched_cues.append("geofencing")
        regions.extend(region for region in _REGIONS if re.search(rf"\b{re.escape(region)}\b", lowered))

    regions = _dedupe(regions)
    return {
        "requires_geofencing": bool(matched_cues),
        "regions": regions,
        "restriction_type": restriction_type if matched_cues else "none",
        "matched_cues": _dedupe(matched_cues),
        "severity": "high" if restriction_type in {"allowlist", "blocklist", "mixed"} else "medium" if matched_cues else "none",
    }


def _collect_matches(patterns: tuple[str, ...], text: str, matched_cues: list[str], regions: list[str]) -> bool:
    found = False
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            found = True
            matched_cues.append("location_access_restriction")
            regions.extend(region for region in _REGIONS if re.search(rf"\b{re.escape(region)}\b", match.group("region").casefold()))
    return found


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
