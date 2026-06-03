"""Detect biometric data requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("biometric_identifiers", "high", (r"\bbiometric\s+(?:data|identifiers?|information)\b",)),
    ("face", "high", (r"\bface\s+recognition\b", r"\bfacial\s+recognition\b", r"\bfaceprint\b")),
    ("fingerprint", "high", (r"\bfingerprints?\s+(?:scan|template|matching|identity)\b", r"\bfingerprint\s+biometrics?\b")),
    ("voice", "high", (r"\bvoiceprints?\b", r"\bvoice\s+biometrics?\b")),
    ("palm", "high", (r"\bpalm\s+scans?\b", r"\bpalmprint\b")),
    ("liveness", "medium", (r"\bliveness\s+detection\b", r"\bbiometric\s+liveness\b")),
    ("consent", "medium", (r"\bbiometric\s+consent\b", r"\bconsent\s+for\s+biometrics?\b")),
    ("retention", "medium", (r"\bbiometric\s+retention\b", r"\bretention\s+of\s+biometric\s+data\b")),
    ("deletion", "medium", (r"\bbiometric\s+deletion\b", r"\bdelete\s+biometric\s+data\b")),
)


def detect_query_biometric_data_requirement(query: str) -> dict[str, Any]:
    matches = _matches(query)
    categories = sorted(dict.fromkeys(match["category"] for match in matches))
    return {"has_biometric_data_requirements": bool(matches), "categories": categories, "matches": matches}


def _matches(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                rows.append({"matched_text": match.group(0), "category": category, "severity": severity, "span": match.span()})
    return sorted(rows, key=lambda row: (row["span"][0], row["category"], row["matched_text"].casefold()))
