"""Detect API schema requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_API_CONTEXT_RE = re.compile(r"\b(?:api|endpoint|request|response|graphql|protobuf|openapi|swagger|json\s+schema)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("openapi", "high", (r"\bopen\s*api\b", r"\bswagger\b")),
    ("json_schema", "medium", (r"\bjson\s+schema\b",)),
    ("graphql_schema", "medium", (r"\bgraphql\s+schema\b",)),
    ("protobuf", "medium", (r"\bprotobuf\b", r"\bprotocol\s+buffers?\b", r"\bproto\s+schema\b")),
    ("request_response_schema", "high", (r"\brequest\s*/\s*response\s+schema\b", r"\brequest\s+and\s+response\s+schema\b", r"\bresponse\s+schema\b", r"\brequest\s+schema\b")),
    ("schema_versioning", "medium", (r"\bschema\s+version(?:ing|s)?\b", r"\bversioned\s+schemas?\b")),
)


def detect_query_api_schema_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match and _has_api_context(category, text):
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_api_schema_requirements": bool(requirements), "requirements": requirements}


def _has_api_context(category: str, text: str) -> bool:
    return category in {"openapi", "json_schema", "graphql_schema", "protobuf"} or bool(_API_CONTEXT_RE.search(text))


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
