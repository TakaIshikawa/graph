"""Detect data minimization requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUE_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("collect_only_necessary", "high", (r"\bcollect\s+only\s+(?:the\s+)?necessary\s+data\b", r"\bonly\s+collect\s+(?:what|data)\s+(?:is\s+)?necessary\b", r"\bminimum\s+necessary\s+data\b")),
    ("pii_minimization", "high", (r"\bminimi[sz]e\s+(?:pii|personal\s+data|personal\s+information)\b", r"\blimit\s+(?:pii|personal\s+data|personal\s+information)\s+collection\b")),
    ("minimize_data", "medium", (r"\bdata\s+minimi[sz]ation\b", r"\bminimi[sz]e\s+(?:the\s+)?data\b", r"\breduce\s+data\s+collection\b")),
    ("limit_collection", "medium", (r"\blimit\s+(?:data\s+)?collection\b", r"\bavoid\s+collecting\s+unnecessary\s+data\b", r"\bdo\s+not\s+collect\s+unnecessary\s+data\b")),
    ("truncate_data", "medium", (r"\btruncat(?:e|ed|ion)\s+(?:data|fields?|payloads?)\b", r"\blimit\s+(?:field|payload)\s+length\b")),
    ("anonymize_data", "medium", (r"\banonymi[sz]e\s+(?:data|records?|users?)\b", r"\bde[-\s]?identify\s+(?:data|records?|users?)\b")),
)


def detect_query_data_minimization_requirement(query: str) -> dict[str, Any]:
    """Return data minimization cues mentioned by a query."""
    text = " ".join(str(query or "").split())
    matched_cues = [
        cue for cue, _severity, patterns in _CUE_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    high_cues = {cue for cue, severity, _patterns in _CUE_SPECS if severity == "high"}
    severity = "high" if any(cue in high_cues for cue in matched_cues) else "medium" if matched_cues else "none"
    return {
        "requires_data_minimization": bool(matched_cues),
        "matched_cues": matched_cues,
        "severity": severity,
    }
