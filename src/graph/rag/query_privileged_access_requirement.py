"""Detect privileged access requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("admin_approval", re.compile(r"\b(?:admin(?:istrator)?\s+approval|approval\s+from\s+an\s+admin|manager\s+approval\s+for\s+admin)\b", re.I), "high"),
    ("admin_role", re.compile(r"\b(?:sudo\s+access|sudoers?|admin\s+role|administrator\s+role|root\s+access)\b", re.I), "high"),
    ("break_glass", re.compile(r"\b(?:break[-\s]?glass|emergency\s+access\s+account|firecall)\b", re.I), "critical"),
    ("jit_access", re.compile(r"\b(?:just[-\s]?in[-\s]?time\s+access|jit\s+access|time[-\s]?bound\s+privileged\s+access)\b", re.I), "high"),
    ("pam", re.compile(r"\b(?:pam|privileged\s+access\s+management|privileged\s+account\s+management)\b", re.I), "high"),
    ("privilege_elevation", re.compile(r"\b(?:privilege\s+elevation|elevat(?:e|ion)\s+privileges?|temporary\s+elevation)\b", re.I), "high"),
)


def detect_query_privileged_access_requirements(query: str) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    rows = []
    for category, pattern, severity in _CATEGORY_SPECS:
        match = pattern.search(normalized)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "severity": severity})
    rows.sort(key=lambda row: row["category"])
    return rows


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
