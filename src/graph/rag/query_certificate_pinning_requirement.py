"""Detect certificate pinning requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("certificate_pinning", (r"\bcertificate\s+pinning\b", r"\bpin\s+(?:the\s+)?certificate\b", r"\bpinned\s+certificates?\b")),
    ("public_key_pinning", (r"\bpublic[-\s]?key\s+pinning\b", r"\bpin\s+(?:the\s+)?public\s+key\b", r"\bspki\s+pinning\b")),
    ("hpkp", (r"\bhpkp\b", r"\bhttp\s+public[-\s]?key\s+pinning\b")),
    ("mobile_pinning", (r"\bmobile\s+(?:certificate\s+)?pinning\b", r"\bios\s+(?:certificate\s+)?pinning\b", r"\bandroid\s+(?:certificate\s+)?pinning\b")),
    ("pin_rotation", (r"\bpin\s+rotation\b", r"\brotate\s+(?:certificate\s+)?pins?\b")),
    ("backup_pins", (r"\bbackup\s+pins?\b", r"\bsecondary\s+pins?\b")),
    ("pinning_bypass", (r"\bpinning\s+bypass\b", r"\bbypass\s+(?:certificate\s+)?pinning\b")),
)
_ROTATION_RE = re.compile(r"\b(?:pin\s+rotation|rotate\s+(?:certificate\s+)?pins?|backup\s+pins?|secondary\s+pins?)\b", re.I)


def detect_query_certificate_pinning_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    requirements = _collect_requirements(text)
    return {
        "has_certificate_pinning_requirements": bool(requirements),
        "requirements": requirements,
        "rotation_sensitive": bool(_ROTATION_RE.search(text)),
    }


def _collect_requirements(text: str) -> list[dict[str, str]]:
    rows = []
    for category, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            rows.append({"category": category, "matched_text": match.group(0)})
    return rows


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
