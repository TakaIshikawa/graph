"""Analyze geographic coverage across retrieved RAG results."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, metadata, result_id, string, value

_LOCATIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("United States", "country", (r"\bUnited States\b", r"\bU\.S\.\b", r"\bUSA\b")),
    ("Canada", "country", (r"\bCanada\b",)),
    ("United Kingdom", "country", (r"\bUnited Kingdom\b", r"\bUK\b", r"\bU\.K\.\b")),
    ("Germany", "country", (r"\bGermany\b",)),
    ("France", "country", (r"\bFrance\b",)),
    ("Japan", "country", (r"\bJapan\b",)),
    ("China", "country", (r"\bChina\b",)),
    ("India", "country", (r"\bIndia\b",)),
    ("Europe", "region", (r"\bEurope\b", r"\bEuropean Union\b", r"\bEU\b")),
    ("Asia", "region", (r"\bAsia\b",)),
    ("Africa", "region", (r"\bAfrica\b",)),
    ("Latin America", "region", (r"\bLatin America\b",)),
    ("Global", "global", (r"\bglobal\b", r"\bworldwide\b", r"\binternational\b")),
)
_GLOBAL_QUERY_RE = re.compile(r"\b(?:global|worldwide|international|cross-country|countries|regions)\b", re.I)
_LOCATION_QUERY_RE = re.compile(r"\b(?:in|for|across)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\b")


def analyze_result_geographic_coverage(query: str, results: Iterable[Any]) -> dict[str, Any]:
    """Return location counts and concentration warnings for location-sensitive queries."""
    normalized_query = " ".join(str(query or "").split())
    rows = []
    counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for index, result in enumerate(results):
        locations = _detect_locations(_result_geo_text(result))
        for location in locations:
            counts[location["location"]] += 1
            type_counts[location["type"]] += 1
        rows.append({"result_id": result_id(result, index), "locations": locations})
    total = len(rows)
    global_query = bool(_GLOBAL_QUERY_RE.search(normalized_query))
    dominant = counts.most_common(1)[0] if counts else None
    warnings = []
    if total == 0:
        warnings.append("no_results")
    if global_query and dominant and dominant[1] / max(total, 1) >= 0.75 and len(counts) <= 2:
        warnings.append("single_region_concentration")
    missing = []
    requested = _requested_locations(normalized_query)
    for location in requested:
        if location not in counts:
            missing.append(location)
    return {
        "result_count": total,
        "location_counts": dict(sorted(counts.items())),
        "coverage_type_counts": dict(sorted(type_counts.items())),
        "results": rows,
        "concentration_warnings": warnings,
        "missing_location_hints": missing,
    }


def _result_geo_text(result: Any) -> str:
    parts = [content_text(result)]
    for key in ("country", "region", "location", "jurisdiction"):
        text = string(value(result, key))
        if text:
            parts.append(text)
    parts.extend(string(item) or "" for item in metadata(result).values())
    return " ".join(parts)


def _detect_locations(text: str) -> list[dict[str, str]]:
    rows = []
    for name, location_type, patterns in _LOCATIONS:
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            rows.append({"location": name, "type": location_type})
    return rows


def _requested_locations(query: str) -> list[str]:
    known = {row["location"] for row in _detect_locations(query) if row["location"] != "Global"}
    for match in _LOCATION_QUERY_RE.finditer(query):
        candidate = match.group(1).strip()
        if candidate.casefold() not in {"global", "worldwide", "countries", "regions"}:
            known.add(candidate)
    return sorted(known)
