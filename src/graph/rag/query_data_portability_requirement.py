"""Detect data portability requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("data_portability", "high", (r"\bdata\s+portability\b", r"\bportable\s+customer\s+data\b")),
    ("export_format", "medium", (r"\bexport\s+formats?\b", r"\b(?:csv|json|xml|parquet)\s+exports?\b", r"\b(?:csv|json|xml|parquet)\b")),
    ("bulk_export", "high", (r"\bbulk\s+exports?\b", r"\bexport\s+all\b", r"\bfull\s+exports?\b")),
    ("customer_data_extraction", "high", (r"\bcustomer\s+data\s+(?:export|extraction|extract)\b", r"\bextract\s+customer\s+data\b")),
    ("offboarding_migration", "high", (r"\boffboarding\s+migration\b", r"\bmigration\s+offboarding\b", r"\bmigrate\s+(?:off|away|out)\b")),
    ("machine_readable_export", "medium", (r"\bmachine[-\s]readable\s+exports?\b", r"\bprogrammatic\s+exports?\b", r"\bstructured\s+exports?\b")),
)


def detect_query_data_portability_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    rows: list[dict[str, Any]] = []
    for category, severity, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"category": category, "severity": severity, "matched_text": match.group(0), "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["category"]))
    return {
        "requires_data_portability": bool(rows),
        "categories": sorted({row["category"] for row in rows}),
        "matches": rows,
    }
