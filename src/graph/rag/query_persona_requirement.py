"""Detect persona requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PERSONAS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("executive", "high", (r"\bfor\s+(?:an?\s+)?executives?\b", r"\bfor\s+(?:the\s+)?c[- ]?suite\b", r"\bfor\s+(?:senior\s+)?leadership\b", r"\bboard(?:\s+of\s+directors)?\b")),
    ("beginner", "medium", (r"\bfor\s+beginners?\b", r"\bfor\s+non[- ]?experts?\b", r"\bfor\s+lay(?:people|persons?|readers?)\b", r"\bin\s+plain\s+english\b")),
    ("clinician", "high", (r"\bfor\s+clinicians?\b", r"\bfor\s+doctors?\b", r"\bfor\s+physicians?\b", r"\bfor\s+nurses?\b")),
    ("developer", "medium", (r"\bfor\s+developers?\b", r"\bfor\s+software\s+engineers?\b", r"\bfor\s+programmers?\b", r"\bdeveloper[- ]focused\b")),
    ("child", "medium", (r"\bfor\s+children\b", r"\bfor\s+kids\b", r"\bfor\s+(?:a\s+)?(?:\d+[- ]?year[- ]?old|child)\b")),
    ("policy maker", "high", (r"\bfor\s+policy\s*makers?\b", r"\bfor\s+policymakers?\b", r"\bfor\s+regulators?\b", r"\bfor\s+legislators?\b")),
)


def detect_query_persona_requirements(query: str) -> list[dict[str, Any]]:
    """Return requested persona constraints in deterministic order."""
    text = " ".join(str(query or "").split())
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for persona, severity, patterns in _PERSONAS:
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match and persona not in seen:
                seen.add(persona)
                rows.append({"persona": persona, "matched_text": match.group(0), "severity": severity})
                break
    return sorted(rows, key=lambda row: (row["persona"], row["matched_text"].casefold()))
