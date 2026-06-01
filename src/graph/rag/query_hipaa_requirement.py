"""Detect HIPAA and protected-health-information requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_HIPAA = (r"\bhipaa\b", r"\bhealth\s+insurance\s+portability\s+and\s+accountability\s+act\b")
_PROTECTED_DATA = (r"\bphi\b", r"\bephi\b", r"\bprotected\s+health\s+information\b", r"\bhealthcare\s+data\b")
_AGREEMENT = (r"\bbusiness\s+associate\s+agreement\b", r"\bbaa\b")
_ENTITY = (r"\bcovered\s+entity\b", r"\bbusiness\s+associate\b")
_OTHER = (r"\bminimum\s+necessary\b",)


def detect_query_hipaa_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    framework = _matches(text, _HIPAA)
    protected = _matches(text, _PROTECTED_DATA)
    agreement = _matches(text, _AGREEMENT)
    entity = _matches(text, _ENTITY)
    other = _matches(text, _OTHER)
    matched = framework + protected + agreement + entity + other
    return {
        "requires_hipaa": bool(matched),
        "matched_phrases": matched,
        "agreement_cues": agreement,
        "protected_data_cues": protected,
        "entity_cues": entity,
        "safeguard_cues": other,
        "confidence": "high" if matched else "none",
    }


def _matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [match.group(0) for pattern in patterns for match in re.finditer(pattern, text, re.I)]
