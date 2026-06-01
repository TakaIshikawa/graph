"""Detect webhook signature verification requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_WEBHOOK_RE = re.compile(r"\bwebhooks?\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("hmac_signature", "high", (r"\bhmac(?:[-\s]sha\d+)?\b", r"\bsignature\s+verification\b", r"\bverify\s+(?:the\s+)?signature\b")),
    ("signing_secret", "high", (r"\bsigning\s+secret\b", r"\bwebhook\s+secret\b")),
    ("timestamp_tolerance", "medium", (r"\btimestamp\s+tolerance\b", r"\btimestamp\s+window\b", r"\bclock\s+skew\b")),
    ("replay_protection", "high", (r"\breplay\s+protection\b", r"\bprevent\s+replay\b", r"\bnonce\b")),
    ("signature_header", "medium", (r"\bx[-\w]*signature\b", r"\bsignature\s+header\b", r"\bstripe-signature\b")),
    ("canonical_payload", "medium", (r"\bcanonical(?:ized)?\s+payload\b", r"\bpayload\s+canonicalization\b", r"\bsigned\s+payload\b")),
)


def detect_query_webhook_signature_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match and _WEBHOOK_RE.search(text):
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_webhook_signature_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
