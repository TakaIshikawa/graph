"""Detect webhook delivery and integration requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(r"\b(?:webhooks?|callback\s+url|api\s+events?|event\s+subscriptions?|integration\s+events?)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("delivery_logs", "medium", (r"\bdelivery\s+logs?\b", r"\bwebhook\s+logs?\b", r"\bdelivery\s+history\b")),
    ("event_types", "medium", (r"\bevent\s+types?\b", r"\bevents?\s+supported\b", r"\bsubscribe\s+to\s+events?\b")),
    ("idempotency", "high", (r"\bidempotenc(?:y|e)\b", r"\bidempotent\b", r"\bdedupe(?:lication)?\b")),
    ("replay", "high", (r"\breplay\b", r"\bredeliver(?:y|ies)?\b")),
    ("retries", "high", (r"\bretr(?:y|ies)\b", r"\bbackoff\b", r"\bfailed\s+deliver(?:y|ies)\b")),
    ("signing_secrets", "high", (r"\bsigning\s+secrets?\b", r"\bwebhook\s+secrets?\b", r"\bsignature\s+verification\b", r"\bhmac\b")),
    ("subscription_verification", "medium", (r"\bsubscription\s+verification\b", r"\bverify\s+(?:the\s+)?subscription\b", r"\bchallenge\s+response\b")),
    ("timeout", "medium", (r"\btimeouts?\b", r"\bresponse\s+deadline\b", r"\b(?:ack|acknowledge)\s+within\b")),
)


def detect_query_webhook_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    rows: list[dict[str, Any]] = []
    if _CONTEXT_RE.search(text):
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                rows.append({"category": category, "matched_text": match.group(0), "severity": severity})
    rows.sort(key=lambda row: row["category"])
    return {"has_webhook_requirement": bool(rows), "requirements": rows}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
