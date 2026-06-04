"""Detect web application firewall requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(
    r"\b(?:waf|web\s+application\s+firewall|managed\s+rules?|owasp\s+crs|bot\s+protection|"
    r"request\s+blocking|rate\s+rules?|virtual\s+patching)\b",
    re.I,
)

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("blocking_mode", "high", (r"\brequest\s+blocking\b", r"\bblocking\s+mode\b", r"\bblock\s+malicious\s+requests?\b")),
    ("bot_protection", "medium", (r"\bbot\s+protection\b", r"\bbot\s+mitigation\b")),
    ("logging", "medium", (r"\bwaf\s+logs?\b", r"\blogging\b", r"\baudit\s+logs?\b")),
    ("managed_service", "medium", (r"\bmanaged\s+waf\b", r"\bmanaged\s+service\b", r"\bcloud\s+waf\b")),
    ("rule_sets", "high", (r"\bmanaged\s+rules?\b", r"\bowasp\s+crs\b", r"\bcore\s+rule\s+set\b", r"\brate\s+rules?\b")),
    ("tuning", "medium", (r"\btuning\b", r"\bfalse\s+positives?\b", r"\brule\s+tuning\b")),
    ("virtual_patching", "high", (r"\bvirtual\s+patching\b", r"\bvirtual\s+patch\b")),
)


def detect_query_waf_requirements(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    if not _CONTEXT_RE.search(normalized):
        return {"has_waf_requirements": False, "requirements": [], "normalized_query": normalized}

    requirements = []
    for category, severity, patterns in _CATEGORY_SPECS:
        match = _first_match(normalized, patterns)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})

    requirements.sort(key=lambda row: row["category"])
    return {
        "has_waf_requirements": bool(requirements),
        "requirements": requirements,
        "normalized_query": normalized,
    }


def _first_match(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
