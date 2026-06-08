"""Detect webhook replay protection requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_WEBHOOK_CONTEXT = re.compile(r"\bwebhooks?\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("replay_protection", "high", (r"\breplay\s+protection\b", r"\bprevent\s+replay\b", r"\breplay\s+attacks?\b")),
    ("timestamp_tolerance", "high", (r"\btimestamp\s+tolerance\b", r"\btimestamp\s+window\b", r"\bclock\s+skew\b")),
    ("nonce", "high", (r"\bnonces?\b", r"\bone[-\s]?time\s+tokens?\b")),
    ("signature_timestamp", "high", (r"\bsignature\s+timestamps?\b", r"\btimestamped\s+signatures?\b")),
    (
        "duplicate_delivery_rejection",
        "medium",
        (r"\breject\s+duplicate\s+deliver(?:y|ies)\b", r"\bduplicate\s+delivery\s+rejection\b", r"\bdeduplicate\s+deliver(?:y|ies)\b"),
    ),
    ("replay_window", "high", (r"\breplay\s+windows?\b", r"\bwindow\s+for\s+replays?\b")),
)


def detect_query_webhook_replay_protection_requirement(query: str) -> dict[str, Any]:
    """Return webhook replay protection requirements mentioned by a query."""
    text = _normalize_query(query)
    requirements = []
    if _WEBHOOK_CONTEXT.search(text):
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_webhook_replay_protection_requirement": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
