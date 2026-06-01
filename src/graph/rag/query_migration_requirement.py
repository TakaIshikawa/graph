"""Detect migration requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("migration", re.compile(r"\b(?:migration|migrate|moving\s+from\s+\w+\s+to\s+\w+|transition\s+from)\b", re.I)),
    ("cutover", re.compile(r"\b(?:cutover|go[-\s]live|switchover|launch\s+window)\b", re.I)),
    ("backfill", re.compile(r"\b(?:backfill|historical\s+data|replay\s+old\s+data)\b", re.I)),
    ("import_export", re.compile(r"\b(?:import/export|import\s+and\s+export|export\s+and\s+import|bulk\s+(?:import|export))\b", re.I)),
    ("legacy", re.compile(r"\b(?:legacy\s+(?:replacement|system|platform)|replace\s+legacy|retire\s+legacy)\b", re.I)),
    ("portability", re.compile(r"\b(?:data\s+portability|portable\s+data|takeout|vendor\s+exit|exportable\s+data)\b", re.I)),
)


def detect_query_migration_requirements(query: str) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    rows = []
    for category, pattern in _CATEGORY_SPECS:
        match = pattern.search(normalized)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["category"]))
    return rows


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").casefold().split())
