"""Detect queries that need a jurisdiction before answering."""

from __future__ import annotations

import re
from typing import Any

_DOMAIN_CUES: dict[str, re.Pattern[str]] = {
    "legal": re.compile(r"\b(?:law|legal|lawsuit|contract|liability|rights|court|statute)\b", re.I),
    "tax": re.compile(r"\b(?:tax|deduction|vat|income tax|sales tax|irs)\b", re.I),
    "regulatory": re.compile(r"\b(?:regulation|compliance|permit|license|rule|policy|requirement)\b", re.I),
    "employment": re.compile(r"\b(?:employment|employee|worker|overtime|minimum wage|termination|leave)\b", re.I),
    "medical_coverage": re.compile(r"\b(?:insurance|medicaid|medicare|coverage|reimbursement|covered)\b", re.I),
}
_JURISDICTIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("United States", re.compile(r"\b(?:US|USA|U\.S\.|United States)\b", re.I)),
    ("EU", re.compile(r"\b(?:EU|European Union)\b", re.I)),
    ("UK", re.compile(r"\b(?:UK|U\.K\.|United Kingdom|England|Scotland|Wales)\b", re.I)),
    ("Canada", re.compile(r"\bCanada\b", re.I)),
    ("California", re.compile(r"\b(?:California|CA)\b")),
    ("New York", re.compile(r"\b(?:New York|NY)\b")),
    ("Texas", re.compile(r"\b(?:Texas|TX)\b")),
    ("Florida", re.compile(r"\b(?:Florida|FL)\b")),
    ("Washington", re.compile(r"\b(?:Washington|WA)\b")),
)
_LOCATION_RE = re.compile(r"\b(?:in|for|within|under)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b")
_GENERIC_LOCATION_RE = re.compile(r"\b(?:my state|my country|where I live|local law|jurisdiction)\b", re.I)


def detect_query_jurisdiction_requirement(query: str) -> dict[str, Any]:
    """Return jurisdiction requirement flags for high-stakes location-specific queries."""
    normalized = " ".join(str(query or "").split())
    domains = [name for name, pattern in _DOMAIN_CUES.items() if pattern.search(normalized)]
    jurisdictions = _detected_jurisdictions(normalized)
    generic_location = bool(_GENERIC_LOCATION_RE.search(normalized))
    requires = bool(domains or generic_location)
    missing = requires and not jurisdictions and not generic_location
    reasons = []
    if domains:
        reasons.append("high_stakes_domain")
    if jurisdictions:
        reasons.append("explicit_jurisdiction")
    if generic_location:
        reasons.append("generic_location_specific_cue")
    if missing:
        reasons.append("missing_jurisdiction")
    return {
        "requires_jurisdiction": requires,
        "detected_jurisdictions": jurisdictions,
        "missing_jurisdiction": missing,
        "domains": domains,
        "reasons": reasons,
    }


def _detected_jurisdictions(query: str) -> list[str]:
    found = {name for name, pattern in _JURISDICTIONS if pattern.search(query)}
    for match in _LOCATION_RE.finditer(query):
        value = match.group(1).strip()
        if value.casefold() not in {"the law", "my state", "my country"}:
            found.add(value)
    return sorted(found)
