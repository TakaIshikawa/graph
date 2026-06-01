"""Detect interoperability requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("open_standards", "high", re.compile(r"\b(?:open\s+standards?|standards[-\s]based|ietf|w3c|iso\s+standard)\b", re.I)),
    ("portability", "high", re.compile(r"\b(?:portability|portable|data\s+portability)\b", re.I)),
    ("backward_compatibility", "high", re.compile(r"\b(?:backward[-\s]compatible|backwards[-\s]compatible|backward\s+compatibility|legacy\s+compatible)\b", re.I)),
    ("data_exportability", "medium", re.compile(r"\b(?:data\s+exportability|exportable\s+data|export\s+data|bulk\s+export)\b", re.I)),
    ("vendor_neutrality", "medium", re.compile(r"\b(?:vendor[-\s]neutral|vendor\s+neutrality|avoid\s+vendor\s+lock[-\s]in|no\s+vendor\s+lock[-\s]in)\b", re.I)),
    ("cross_platform", "medium", re.compile(r"\b(?:cross[-\s]platform|multi[-\s]platform|works\s+across\s+platforms?)\b", re.I)),
)


def detect_query_interoperability_requirements(query: str) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    rows = []
    for category, severity, pattern in _CATEGORY_SPECS:
        match = pattern.search(normalized)
        if match:
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity, "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["category"]))
    return rows


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").casefold().split())
