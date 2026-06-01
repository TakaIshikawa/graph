"""Detect data-retention requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("archive", (r"\barchiv(?:e|al|ing)\b", r"\bcold\s+storage\b")),
    ("deletion", (r"\bdeletion\s+windows?\b", r"\bdelete\s+after\b", r"\berasure\b")),
    ("purge", (r"\bpurge\s+polic(?:y|ies)\b", r"\bpurge\s+after\b", r"\bdata\s+purge\b")),
    ("recordkeeping", (r"\brecordkeeping\b", r"\brecord\s+keeping\b", r"\brecords?\s+retention\b")),
    ("retention", (r"\bretention\s+periods?\b", r"\bdata\s+retention\b", r"\bretain\s+for\b")),
)


def detect_query_data_retention_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if not text:
        return []

    rows: list[dict[str, Any]] = []
    for category, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category})
    return sorted(rows, key=lambda row: row["category"])
