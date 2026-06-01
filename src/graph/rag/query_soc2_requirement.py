"""Detect SOC 2 evidence requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SOC2_PATTERNS = (r"\bsoc\s*2\b", r"\bsoc\s*ii\b", r"\bservice\s+organization\s+control\s+2\b")
_REPORT_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("type_ii", (r"\btype\s*ii\b", r"\btype\s*2\b", r"\btype\s*2\s+report\b")),
    ("type_i", (r"\btype\s*i\b", r"\btype\s*1\b")),
    ("audit_report", (r"\baudit\s+reports?\b", r"\bsoc\s*2\s+reports?\b")),
)
_TRUST_SERVICE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("trust_services_criteria", (r"\btrust\s+services?\s+criteria\b", r"\btsc\b")),
    ("security", (r"\bsecurity\b",)),
    ("availability", (r"\bavailability\b",)),
    ("confidentiality", (r"\bconfidentiality\b",)),
    ("processing_integrity", (r"\bprocessing\s+integrity\b",)),
    ("privacy", (r"\bprivacy\b",)),
)


def detect_query_soc2_requirement(query: str) -> dict[str, Any]:
    """Return SOC 2 report and trust-services cues requested by a query."""
    text = " ".join(str(query or "").split())
    soc2_matches = _matches(text, _SOC2_PATTERNS)
    report_types = [name for name, patterns in _REPORT_TYPES if _matches(text, patterns)]
    trust_service_cues = [name for name, patterns in _TRUST_SERVICE_CUES if _matches(text, patterns)]
    requires_soc2 = bool(soc2_matches)
    return {
        "requires_soc2": requires_soc2,
        "matched_phrases": soc2_matches,
        "report_types": report_types,
        "trust_service_cues": trust_service_cues,
        "recommendations": ["retrieve SOC 2 audit report evidence"] if requires_soc2 else [],
        "confidence": "high" if requires_soc2 else "none",
    }


def _matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    found = [match.group(0) for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return sorted(dict.fromkeys(found), key=lambda item: (item.casefold(), item))
