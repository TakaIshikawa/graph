"""Detect break-glass access requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("break_glass_access", (r"\bbreak[-\s]glass\s+access\b", r"\bbreakglass\s+access\b")),
    ("emergency_access", (r"\bemergency\s+access\b", r"\bemergency\s+login\b")),
    ("emergency_admin", (r"\bemergency\s+admin\b", r"\bemergency\s+administrator\b")),
    ("privileged_emergency_account", (r"\bprivileged\s+emergency\s+accounts?\b", r"\bemergency\s+privileged\s+accounts?\b")),
    ("approval", (r"\bapproval\s+for\s+(?:break[-\s]glass|emergency)\b", r"\bapproved\s+(?:break[-\s]glass|emergency)\s+access\b")),
    ("monitoring", (r"\bmonitor(?:ing)?\s+(?:break[-\s]glass|emergency)\s+access\b", r"\bbreak[-\s]glass\s+monitoring\b")),
    ("post_use_review", (r"\bpost[-\s]use\s+review\b", r"\breview\s+after\s+use\b", r"\bafter[-\s]action\s+review\b")),
)


def detect_query_break_glass_requirements(query: str) -> dict[str, Any]:
    """Return break-glass access requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    requirements = [
        requirement
        for requirement, patterns in _REQUIREMENT_SPECS
        if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    return {
        "has_break_glass_requirements": bool(requirements),
        "requirements": requirements,
        "post_use_review_required": "post_use_review" in requirements,
    }
