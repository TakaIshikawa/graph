"""Detect encryption-in-transit requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tls", (r"\btls\b", r"\btransport\s+layer\s+security\b")),
    ("https", (r"\bhttps\b", r"\bencrypted\s+http\b")),
    ("mtls", (r"\bmtls\b", r"\bmutual\s+tls\b")),
    ("certificate_validation", (r"\bcertificate\s+validation\b", r"\bvalidate\s+(?:tls\s+)?certificates?\b", r"\bcertificate\s+chain\b")),
    ("cipher_suite", (r"\bcipher\s+suites?\b", r"\bTLS_[A-Z0-9_]+\b")),
    ("forward_secrecy", (r"\bforward\s+secrecy\b", r"\bpfs\b", r"\bperfect\s+forward\s+secrecy\b")),
    ("minimum_tls_version", (r"\bminimum\s+tls\s+(?:version\s+)?\d+(?:\.\d+)?\b", r"\btls\s+1\.[23]\s+(?:or\s+(?:higher|newer)|minimum|required)\b")),
)
_VALUES = (
    r"\bTLS\s+1\.[0-3]\b",
    r"\bmTLS\b",
    r"\bHTTPS\b",
    r"\bTLS_[A-Z0-9_]+\b",
    r"\b[A-Z]+-GCM-[A-Z0-9-]+\b",
)
_AT_REST_ONLY_RE = re.compile(r"\b(?:at[-\s]?rest|disk|database|storage)\s+encryption\b", re.I)


def detect_query_encryption_in_transit_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    cue_categories = _matched_categories(text)
    values = _extract_values(text)
    if _AT_REST_ONLY_RE.search(text) and not cue_categories and not values:
        cue_categories = []
    return {
        "requires_encryption_in_transit": bool(cue_categories or values),
        "cue_categories": cue_categories,
        "protocol_values": values,
    }


def _matched_categories(text: str) -> list[str]:
    return [category for category, patterns in _CUES if any(re.search(pattern, text, re.I) for pattern in patterns)]


def _extract_values(text: str) -> list[str]:
    values: list[tuple[int, str]] = []
    for pattern in _VALUES:
        for match in re.finditer(pattern, text, re.I):
            values.append((match.start(), match.group(0)))
    return list(dict.fromkeys(value for _pos, value in sorted(values)))


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.split())
