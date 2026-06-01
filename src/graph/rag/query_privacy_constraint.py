"""Detect privacy and data-handling constraints in user queries."""

from __future__ import annotations

import re
from typing import Any

_CUES = (
    ("do not store", "privacy", re.compile(r"\bdo not store\b|\bdon't store\b", re.I), 0.35),
    ("anonymize", "redact", re.compile(r"\banonymi[sz]e\b|de[-\s]?identify", re.I), 0.3),
    ("redact personal info", "redact", re.compile(r"\bredact\b|personal info|pii", re.I), 0.3),
    ("local only", "local", re.compile(r"\blocal only\b|\bon-device\b", re.I), 0.35),
    ("private", "privacy", re.compile(r"\bprivate\b|\bconfidential\b", re.I), 0.25),
    ("avoid external APIs", "external", re.compile(r"\bavoid external (?:apis|services)\b|\bno external (?:apis|services)\b", re.I), 0.35),
    ("can share publicly", "public", re.compile(r"\bcan share publicly\b|\bpublic(?:ly)? ok\b", re.I), 0.25),
)


def detect_query_privacy_constraints(query: str) -> dict[str, Any]:
    text = str(query or "")
    matched = [(label, kind, weight) for label, kind, pattern, weight in _CUES if pattern.search(text)]
    kinds = {kind for _label, kind, _weight in matched}
    confidence = min(1.0, sum(weight for _label, _kind, weight in matched))
    return {
        "has_privacy_constraint": bool(kinds - {"public"}),
        "local_only": "local" in kinds,
        "redact_pii": "redact" in kinds,
        "avoid_external_services": "external" in kinds,
        "public_ok": "public" in kinds,
        "matched_terms": [label for label, _kind, _weight in matched],
        "confidence": round(confidence, 2),
    }
