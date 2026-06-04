"""Detect data loss prevention requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("dlp", "high", (r"\bdlp\b", r"\bdata\s+loss\s+prevention\b")),
    (
        "exfiltration_prevention",
        "high",
        (
            r"\b(?:prevent|block|stop|detect)\s+(?:data\s+)?exfiltration\b",
            r"\bexfiltration\s+prevention\b",
            r"\bdata\s+exfiltration\s+controls?\b",
        ),
    ),
    (
        "sensitive_data_leakage",
        "high",
        (
            r"\b(?:sensitive|confidential|restricted|regulated|pii)\s+data\s+leak(?:age|s)?\b",
            r"\bprevent\s+(?:sensitive|confidential|restricted|regulated|pii)\s+data\s+(?:leakage|leaks?)\b",
            r"\b(?:block|stop)\s+(?:sensitive|confidential|restricted|regulated|pii)\s+data\s+(?:leaving|sharing|egress)\b",
        ),
    ),
    (
        "clipboard_blocking",
        "medium",
        (
            r"\b(?:block|disable|restrict|prevent)\s+(?:copy(?:ing)?|paste|clipboard)\b",
            r"\bclipboard\s+(?:blocking|controls?|restriction|prevention)\b",
            r"\bcopy[-\s]?paste\s+(?:blocking|controls?|restriction|prevention)\b",
        ),
    ),
    (
        "download_blocking",
        "medium",
        (
            r"\b(?:block|disable|restrict|prevent)\s+downloads?\b",
            r"\bdownload\s+(?:blocking|controls?|restriction|prevention)\b",
            r"\bfile\s+download\s+(?:blocking|controls?|restriction|prevention)\b",
        ),
    ),
    (
        "content_inspection",
        "medium",
        (
            r"\bcontent\s+inspection\b",
            r"\binspect\s+(?:content|files?|uploads?|messages?|attachments?)\b",
            r"\bscan\s+(?:content|files?|uploads?|messages?|attachments?)\s+for\s+(?:sensitive|confidential|restricted|regulated|pii)\b",
        ),
    ),
)


def detect_query_dlp_requirement(query: str) -> dict[str, Any]:
    """Return DLP requirement signals mentioned by a query."""
    text = _normalize_query(query)
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append(
                {
                    "category": category,
                    "matched_text": match.group(0),
                    "severity": severity,
                    "evidence_terms": _evidence_terms(match.group(0)),
                }
            )
    requirements.sort(key=lambda row: row["category"])
    return {
        "requires_dlp": bool(requirements),
        "classification": "dlp_requirement" if requirements else "unrelated",
        "requirements": requirements,
        "evidence_terms": sorted({term for row in requirements for term in row["evidence_terms"]}),
    }


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _evidence_terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
