"""Detect structured data format requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_FORMAT_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("csv", re.compile(r"\b(?:csv|comma[-\s]separated(?:\s+values)?)\b", re.I)),
    ("json", re.compile(r"\b(?:json|jsonl|ndjson)\b", re.I)),
    ("xml", re.compile(r"\bxml\b", re.I)),
    ("yaml", re.compile(r"\b(?:yaml|yml)\b", re.I)),
    ("sql", re.compile(r"\b(?:sql|select\s+statement|database\s+query)\b", re.I)),
    ("parquet", re.compile(r"\bparquet\b", re.I)),
    ("spreadsheet", re.compile(r"\b(?:spreadsheet|excel|xlsx?|google\s+sheets?)\b", re.I)),
    ("api_response", re.compile(r"\b(?:api\s+(?:response|payload)|rest\s+response|graphql\s+response|endpoint\s+response|response\s+body|webhook\s+payload)\b", re.I)),
    ("schema", re.compile(r"\b(?:json\s+schema|openapi(?:\s+schema)?|swagger|data\s+model|database\s+schema|table\s+schema|schema)\b", re.I)),
    ("table", re.compile(r"\b(?:table|tabular|rows\s+and\s+columns)\b", re.I)),
    ("machine_readable", re.compile(r"\b(?:machine[-\s]readable|structured\s+data|parseable|parsable|programmatic(?:ally)?\s+(?:output|readable)|computer[-\s]readable)\b", re.I)),
)
_SCHEMA_FORMATS = {"api_response", "schema"}
_MACHINE_READABLE_FORMATS = {"machine_readable"}


def detect_query_data_format_requirement(query: str) -> dict[str, Any]:
    """Return requested data formats and structured retrieval recommendations."""
    normalized = _normalize_query(query)
    cues = _collect_cues(normalized)
    requested_formats = _requested_formats(cues)
    schema_cues = [cue for cue in cues if cue["type"] in _SCHEMA_FORMATS]
    machine_readable_cues = [cue for cue in cues if cue["type"] in _MACHINE_READABLE_FORMATS]
    requires_structured_data = bool(cues)
    return {
        "requires_structured_data": requires_structured_data,
        "requested_formats": requested_formats,
        "schema_cues": schema_cues,
        "machine_readable_cues": machine_readable_cues,
        "recommendations": _recommendations(requested_formats, schema_cues, machine_readable_cues),
        "confidence": _confidence(requested_formats, schema_cues, machine_readable_cues),
        "normalized_query": normalized,
    }


def _collect_cues(normalized_query: str) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for kind, pattern in _FORMAT_SPECS:
        for match in pattern.finditer(normalized_query):
            cues.append({"type": kind, "cue": match.group(0).strip(), "span": [match.start(), match.end()]})
    cues.sort(key=lambda row: (row["span"][0], row["span"][1], row["type"]))
    return cues


def _requested_formats(cues: list[dict[str, Any]]) -> list[str]:
    seen = {cue["type"] for cue in cues}
    return [kind for kind, _pattern in _FORMAT_SPECS if kind in seen]


def _recommendations(
    requested_formats: list[str],
    schema_cues: list[dict[str, Any]],
    machine_readable_cues: list[dict[str, Any]],
) -> list[str]:
    recommendations = []
    if requested_formats:
        recommendations.append("prefer_sources_with_structured_or_exportable_data")
    if schema_cues:
        recommendations.append("retrieve_schema_api_or_contract_documentation")
    if machine_readable_cues:
        recommendations.append("prioritize_machine_readable_sources_over_narrative_summaries")
    if any(format_name in requested_formats for format_name in ("csv", "json", "xml", "yaml", "parquet", "spreadsheet")):
        recommendations.append("preserve_field_names_types_and_units_from_source")
    return recommendations


def _confidence(
    requested_formats: list[str],
    schema_cues: list[dict[str, Any]],
    machine_readable_cues: list[dict[str, Any]],
) -> float:
    explicit_formats = set(requested_formats) - {"table", "schema", "api_response", "machine_readable"}
    if explicit_formats and (schema_cues or machine_readable_cues):
        return 0.95
    if explicit_formats:
        return 0.85
    if schema_cues or machine_readable_cues:
        return 0.75
    if requested_formats:
        return 0.45
    return 0.0


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").casefold().split())
