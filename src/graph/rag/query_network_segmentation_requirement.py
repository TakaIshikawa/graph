"""Detect network segmentation requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("network_segmentation", (r"\bnetwork\s+segmentation\b", r"\bsegmented\s+network\b")),
    ("microsegmentation", (r"\bmicro-?segmentation\b",)),
    ("subnet_isolation", (r"\bsubnet\s+isolation\b", r"\bisolate\s+subnets?\b")),
    ("east_west_controls", (r"\beast-west\s+traffic\b", r"\beast\s+west\s+traffic\s+controls?\b")),
    ("private_networking", (r"\bprivate\s+networking\b", r"\bprivate\s+network\s+access\b")),
    ("zero_trust_boundaries", (r"\bzero-?trust\s+network\s+boundar(?:y|ies)\b", r"\bnetwork\s+boundar(?:y|ies)\b")),
)


def detect_query_network_segmentation_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    categories = [category for category, patterns in _CATEGORIES if _first_match(patterns, text)]
    return {"requires_network_segmentation": bool(categories), "cue_categories": categories}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    text = " ".join(str(query or "").split())
    if not text:
        raise ValueError("query must not be empty")
    return text
