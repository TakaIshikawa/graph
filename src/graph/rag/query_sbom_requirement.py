"""Detect SBOM requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("sbom", "high", re.compile(r"\b(?:sbom|software\s+bill\s+of\s+materials?)\b", re.I)),
    ("spdx", "high", re.compile(r"\bspdx\b", re.I)),
    ("cyclonedx", "high", re.compile(r"\bcyclonedx\b", re.I)),
    ("component_inventory", "high", re.compile(r"\b(?:component\s+inventory|inventory\s+of\s+components|software\s+component\s+list)\b", re.I)),
    ("package_provenance", "medium", re.compile(r"\b(?:package\s+provenance|dependency\s+provenance|provenance\s+for\s+packages?)\b", re.I)),
    ("transitive_dependency_visibility", "medium", re.compile(r"\b(?:transitive\s+dependenc(?:y|ies)\s+(?:visibility|list|inventory)|visibility\s+into\s+transitive\s+dependenc(?:y|ies))\b", re.I)),
)


def detect_query_sbom_requirements(query: str) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    rows = []
    for category, severity, pattern in _CATEGORY_SPECS:
        match = pattern.search(normalized)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "severity": severity, "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["category"]))
    return rows


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
