"""Detect offline-access requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("air_gapped", (r"\bair[-\s]?gapped\b",)),
    ("disconnected", (r"\bdisconnected\b", r"\bwithout\s+connectivity\b")),
    ("local_only", (r"\blocal[-\s]?only\b", r"\bon[-\s]?device\b", r"\bon[-\s]?prem(?:ises)?\s+only\b")),
    ("no_internet", (r"\bno\s+internet\b", r"\bwithout\s+(?:an\s+)?internet\b", r"\bno\s+network\b")),
    ("offline", (r"\boffline\b", r"\bwork\s+offline\b")),
)
_STRICT = {"air_gapped", "local_only", "no_internet"}


def detect_query_offline_access_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    found = [
        name
        for name, patterns in _SIGNALS
        if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    return {
        "requires_offline_access": bool(found),
        "signals": found,
        "strict_offline_required": any(name in _STRICT for name in found),
    }
