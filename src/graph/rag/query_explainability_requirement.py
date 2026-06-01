"""Detect explainability requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("explainability", "high", re.compile(r"\b(?:explainability|explainable\s+(?:ai|model|recommendations?|outputs?)|must\s+explain|needs?\s+to\s+explain)\b", re.I)),
    ("interpretability", "high", re.compile(r"\b(?:interpretability|interpretable\s+(?:model|results?|outputs?|reasoning)|human[-\s]interpretable)\b", re.I)),
    ("rationale", "medium", re.compile(r"\b(?:(?:provide|include|show|surface)\s+(?:a\s+)?rationale|rationale\s+for|reasoning\s+behind|decision\s+rationale)\b", re.I)),
    ("decision_reasons", "medium", re.compile(r"\b(?:decision\s+reasons?|reasons?\s+for\s+(?:the\s+)?decision|why\s+(?:the\s+)?decision\s+was\s+made)\b", re.I)),
    ("model_transparency", "high", re.compile(r"\b(?:model\s+transparency|transparent\s+(?:model|reasoning|decisioning)|transparency\s+into\s+(?:the\s+)?model)\b", re.I)),
    ("recommendation_rationale", "medium", re.compile(r"\b(?:why\s+(?:a\s+)?recommendation\s+was\s+made|reasons?\s+for\s+(?:the\s+)?recommendation|recommendation\s+rationale)\b", re.I)),
)


def detect_query_explainability_requirements(query: str) -> list[dict[str, Any]]:
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
