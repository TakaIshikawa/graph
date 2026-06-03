"""Detect encryption requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SECURITY_CONTEXT_RE = re.compile(
    r"\b(?:encrypt(?:ion|ed)?|tls|ssl|https|kms|keys?|csek|cmk|byok|customer[-\s]?managed|rotate|rotation|at\s+rest|in\s+transit)\b",
    re.I,
)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("customer_managed_keys", "high", (r"\bcustomer[-\s]?managed\s+keys?\b", r"\bcmks?\b", r"\bbring\s+your\s+own\s+keys?\b", r"\bbyok\b", r"\bcustomer[-\s]?supplied\s+encryption\s+keys?\b", r"\bcsek\b")),
    ("encryption_at_rest", "high", (r"\bencryption\s+at\s+rest\b", r"\bencrypt(?:ed)?\s+(?:data|storage|database|files?|objects?|backups?)\s+at\s+rest\b", r"\bat[-\s]?rest\s+encryption\b")),
    ("encryption_in_transit", "high", (r"\bencryption\s+in\s+transit\b", r"\bencrypt(?:ed)?\s+(?:data|traffic|connections?)\s+in\s+transit\b", r"\bin[-\s]?transit\s+encryption\b")),
    ("key_rotation", "medium", (r"\bkey\s+rotation\b", r"\brotate\s+(?:encryption\s+)?keys?\b", r"\b(?:automatic|scheduled)\s+key\s+rotation\b")),
    ("kms", "medium", (r"\bkms\b", r"\bkey\s+management\s+services?\b", r"\bcloud\s+key\s+management\b")),
    ("tls", "high", (r"\btls\s*(?:1\.[23])?\b", r"\bmutual\s+tls\b", r"\bmtls\b", r"\bhttps\s+only\b", r"\bssl\/tls\b")),
)


def detect_query_encryption_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    requirements = []
    if _SECURITY_CONTEXT_RE.search(text):
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_encryption_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
