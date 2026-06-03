"""Detect encryption key management requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"\bencrypt(?:ion|ed|ing)?\b",
    r"\bcrypto(?:graphic|graphy)?\b",
    r"\bcloud\b",
    r"\bsecurity\b",
    r"\bcompliance\b",
    r"\bdata\s+protection\b",
    r"\bsecrets?\b",
    r"\btenant\b",
)
_PHYSICAL_KEY_PATTERNS: tuple[str, ...] = (
    r"\bdoor\s+keys?\b",
    r"\bcar\s+keys?\b",
    r"\bhouse\s+keys?\b",
    r"\boffice\s+keys?\b",
    r"\bphysical\s+keys?\b",
    r"\bkey\s+card\b",
)
_REQUIREMENT_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("customer_managed_key", "high", (r"\bcmk\b", r"\bcustomer[-\s]managed\s+keys?\b", r"\bcustomer\s+managed\s+encryption\s+keys?\b")),
    ("bring_your_own_key", "high", (r"\bbyok\b", r"\bbring\s+your\s+own\s+keys?\b")),
    ("hsm_kms", "high", (r"\bkms\b", r"\bhsm\b", r"\bhardware\s+security\s+modules?\b", r"\bkey\s+management\s+service\b")),
    ("key_rotation", "medium", (r"\bkey\s+rotation\b", r"\brotate\s+(?:the\s+)?(?:encryption\s+)?keys?\b", r"\brotation\s+of\s+(?:encryption\s+)?keys?\b")),
    ("envelope_encryption", "medium", (r"\benvelope\s+encryption\b",)),
    ("key_escrow", "medium", (r"\bkey\s+escrow\b", r"\bescrow\s+(?:of\s+)?(?:encryption\s+)?keys?\b")),
)


def detect_query_encryption_key_management_requirements(query: str) -> dict[str, Any]:
    """Return encryption key-management requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    requirements = _requirements(text)
    return {
        "has_encryption_key_management_requirements": bool(requirements),
        "requirements": requirements,
    }


def _requirements(text: str) -> list[dict[str, Any]]:
    if not text or (_looks_physical_key_only(text) and not _has_context(text)):
        return []

    rows: list[dict[str, Any]] = []
    for category, severity, patterns in _REQUIREMENT_SPECS:
        match = _first_match(patterns, text)
        if match and (_is_strong_crypto_category(category) or _has_context(text)):
            rows.append(
                {
                    "category": category,
                    "severity": severity,
                    "matched_text": match.group(0),
                    "span": (match.start(), match.end()),
                }
            )
    return sorted(rows, key=lambda row: (row["severity"] != "high", row["category"]))


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _has_context(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in _CONTEXT_PATTERNS)


def _looks_physical_key_only(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in _PHYSICAL_KEY_PATTERNS)


def _is_strong_crypto_category(category: str) -> bool:
    return category in {"customer_managed_key", "bring_your_own_key", "hsm_kms", "envelope_encryption"}
