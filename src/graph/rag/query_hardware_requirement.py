"""Detect hardware prerequisite requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUE_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cpu", re.compile(r"\b(?:cpu|processor|x86_64|arm64|apple\s+silicon)\b", re.I)),
    ("gpu", re.compile(r"\b(?:gpu|graphics\s+card|nvidia\s+a100)\b", re.I)),
    ("ram", re.compile(r"\b(?:ram|memory)\b", re.I)),
    ("storage", re.compile(r"\b(?:storage|ssd|disk\s+space)\b", re.I)),
    ("device_model", re.compile(r"\b(?:device\s+model|hardware\s+model|appliance\s+model)\b", re.I)),
    ("accelerator", re.compile(r"\b(?:accelerator|tpu|npu|hsm)\b", re.I)),
    ("minimum_hardware", re.compile(r"\b(?:minimum\s+hardware|hardware\s+requirements?|hardware\s+prerequisites?)\b", re.I)),
    ("on_prem_appliance", re.compile(r"\b(?:on[-\s]?prem(?:ises)?\s+appliance|physical\s+appliance)\b", re.I)),
)

_VALUE_PATTERN = re.compile(
    r"\b(?:\d+\s*(?:gb|tb)\s+(?:ram|memory|storage|ssd)|nvidia\s+a100|apple\s+silicon|x86_64|arm64|ssd|tpm\s*2\.0)\b",
    re.I,
)


def detect_query_hardware_requirement(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    cue_matches = _cue_matches(normalized)
    return {
        "requires_hardware_requirement": bool(cue_matches),
        "cue_categories": [match["category"] for match in cue_matches],
        "matched_cues": cue_matches,
        "hardware_values": _hardware_values(normalized),
        "normalized_query": normalized,
    }


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.split())


def _cue_matches(normalized_query: str) -> list[dict[str, Any]]:
    rows = []
    for category, pattern in _CUE_SPECS:
        match = pattern.search(normalized_query)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["category"]))
    return rows


def _hardware_values(normalized_query: str) -> list[str]:
    seen: set[str] = set()
    values = []
    for match in _VALUE_PATTERN.finditer(normalized_query):
        value = match.group(0)
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values
