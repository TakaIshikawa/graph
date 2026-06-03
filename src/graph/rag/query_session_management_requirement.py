"""Detect session management requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_AUTH_SESSION_CONTEXT_RE = re.compile(r"\b(?:(?:auth(?:entication)?|login|log\s*in|sign[-\s]?in|user|account)\s+\w*\s*sessions?|sessions?\s+\w*\s*(?:auth(?:entication)?|login|user|account)|cookie[-\s]?backed\s+sessions?|remember\s+me)\b", re.I)
_BROWSER_ONLY_RE = re.compile(r"\bbrowser\s+session\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("absolute_timeout", "high", (r"\babsolute\s+(?:session\s+)?timeout\b", r"\bmaximum\s+session\s+(?:age|lifetime)\b", r"\bsession\s+expires?\s+after\b")),
    ("concurrent_sessions", "medium", (r"\bconcurrent\s+sessions?\b", r"\bsimultaneous\s+logins?\b", r"\blimit\s+(?:active\s+)?sessions?\b")),
    ("device_logout", "medium", (r"\blog\s*out\s+(?:all\s+)?devices?\b", r"\bsign\s*out\s+(?:all\s+)?devices?\b", r"\bdevice\s+logout\b")),
    ("idle_timeout", "high", (r"\bidle\s+(?:session\s+)?timeout\b", r"\binactivity\s+timeout\b", r"\btimeout\s+inactive\s+sessions?\b")),
    ("remember_me", "medium", (r"\bremember\s+me\b", r"\bpersistent\s+login\b", r"\btrusted\s+devices?\b")),
    ("secure_cookie", "high", (r"\bsecure\s+(?:session\s+)?cookies?\b", r"\bhttponly\s+(?:session\s+)?cookies?\b", r"\bsamesite\s+(?:session\s+)?cookies?\b", r"\bcookie[-\s]?backed\s+sessions?\b")),
    ("session_revocation", "high", (r"\bsession\s+revocation\b", r"\brevoke\s+(?:user\s+)?sessions?\b", r"\binvalidate\s+(?:active\s+)?sessions?\b")),
)


def detect_query_session_management_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    has_context = bool(_AUTH_SESSION_CONTEXT_RE.search(text)) and not (_BROWSER_ONLY_RE.search(text) and not re.search(r"\b(?:auth|login|account|user)\b", text, re.I))
    requirements = []
    if has_context:
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_session_management_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
