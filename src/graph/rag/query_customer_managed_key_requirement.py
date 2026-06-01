"""Detect customer-managed key requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cmk", (r"\bcmk\b", r"\bcustomer[-\s]managed\s+keys?\b")),
    ("byok", (r"\bbyok\b", r"\bbring\s+your\s+own\s+keys?\b")),
    ("ekm", (r"\bekm\b", r"\bexternal\s+key\s+management\b")),
    ("key_custody", (r"\bkey\s+custody\b", r"\bcustody\s+of\s+(?:the\s+)?keys?\b")),
    ("tenant_key", (r"\btenant\s+keys?\b", r"\btenant[-\s]specific\s+keys?\b")),
)


def detect_query_customer_managed_key_requirements(query: str) -> dict[str, Any]:
    """Return customer-managed key requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    requirements = [
        requirement
        for requirement, patterns in _REQUIREMENT_SPECS
        if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    return {
        "has_customer_managed_key_requirements": bool(requirements),
        "requirements": requirements,
        "external_key_sensitive": "ekm" in requirements,
    }
