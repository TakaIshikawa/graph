"""Detect device posture requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("compliant_device", re.compile(r"\b(?:compliant\s+device|device\s+compliance|compliance\s+posture)\b", re.I), "high"),
    ("disk_encryption", re.compile(r"\b(?:disk\s+encryption|full[-\s]?disk\s+encrypted|filevault|bitlocker)\b", re.I), "high"),
    ("edr_mdm", re.compile(r"\b(?:edr|mdm|endpoint\s+detection|mobile\s+device\s+management|device\s+management\s+agent)\b", re.I), "high"),
    ("jailbreak_root", re.compile(r"\b(?:jailbreak|jailbroken|rooted\s+device|root\s+detection|tamper(?:ed)?\s+device)\b", re.I), "high"),
    ("managed_device", re.compile(r"\b(?:managed\s+devices?|enrolled\s+devices?|company[-\s]?managed\s+(?:laptops?|devices?|endpoints?))\b", re.I), "high"),
    ("os_version", re.compile(r"\b(?:os\s+version|minimum\s+os|operating\s+system\s+version|patch\s+level)\b", re.I), "medium"),
)


def detect_query_device_posture_requirements(query: str) -> list[dict[str, Any]]:
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
