"""Detect rollout strategy requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("pilot", "medium", re.compile(r"\b(?:pilot|proof\s+of\s+concept|poc)\b", re.I)),
    ("phased", "medium", re.compile(r"\b(?:phased\s+rollout|staged\s+rollout|gradual\s+rollout|roll\s+out\s+in\s+phases)\b", re.I)),
    ("canary", "high", re.compile(r"\bcanary(?:\s+(?:release|deployment|rollout))?\b", re.I)),
    ("blue_green", "high", re.compile(r"\bblue[-\s]green(?:\s+(?:deployment|release))?\b", re.I)),
    ("migration_window", "high", re.compile(r"\b(?:migration\s+window|maintenance\s+window|release\s+window)\b", re.I)),
    ("feature_flag", "high", re.compile(r"\b(?:feature\s+flags?|feature\s+toggles?|flagged\s+rollout)\b", re.I)),
    ("beta", "medium", re.compile(r"\b(?:beta|private\s+beta|public\s+beta|early\s+access)\b", re.I)),
    ("adoption_plan", "medium", re.compile(r"\b(?:adoption\s+plan|change\s+management|enablement\s+plan)\b", re.I)),
)


def detect_query_rollout_strategy_requirements(query: str) -> list[dict[str, Any]]:
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
