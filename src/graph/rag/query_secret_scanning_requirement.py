"""Detect secret-scanning requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(
    r"\bsecret\s+scann?ing\b"
    r"|\bcredential\s+scann?ing\b"
    r"|\btoken\s+leak\s+detection\b"
    r"|\bexposed\s+api\s+keys?\b"
    r"|\bpre[-\s]?commit\s+scann?ing\b"
    r"|\brepositor(?:y|ies)\s+scann?ing\b"
    r"|\bpush\s+protection\b",
    re.I,
)

_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "scan_surface",
        re.compile(
            r"\brepositor(?:y|ies)\s+scann?ing\b|\bscan\s+(?:git\s+)?repos(?:itor(?:y|ies))?\b|\bpre[-\s]?commit\s+scann?ing\b|\bcommit\s+scann?ing\b",
            re.I,
        ),
    ),
    (
        "prevention",
        re.compile(r"\bpush\s+protection\b|\bblock(?:ing)?\s+(?:secret|credential|token|api\s+key)\s+commits?\b|\bpre[-\s]?commit\s+hooks?\b", re.I),
    ),
    (
        "detection",
        re.compile(r"\bsecret\s+scann?ing\b|\bcredential\s+scann?ing\b|\btoken\s+leak\s+detection\b|\bexposed\s+api\s+keys?\b|\bleaked\s+tokens?\b", re.I),
    ),
    (
        "alerting",
        re.compile(r"\balerts?\b|\bnotifications?\b|\bnotify\b|\bsiem\b|\bwebhooks?\b", re.I),
    ),
    (
        "remediation",
        re.compile(r"\bremediat(?:e|ion)\b|\brevok(?:e|ing|ed)\b|\brotate\b|\binvalidat(?:e|ion|ing)\b|\bremove\s+(?:the\s+)?secret\b", re.I),
    ),
    (
        "exceptions",
        re.compile(r"\bexceptions?\b|\ballowlists?\b|\bignore\s+rules?\b|\bfalse\s+positives?\b|\bsuppress(?:ion|ions)?\b", re.I),
    ),
)


def detect_query_secret_scanning_requirements(query: str) -> dict[str, Any]:
    """Return secret-scanning requirement categories mentioned by a query."""

    text = _normalize_query(query)
    if not _CONTEXT_RE.search(text):
        return {"has_secret_scanning_requirements": False, "requirements": []}

    rows: list[dict[str, Any]] = []
    for category, pattern in _SPECS:
        match = pattern.search(text)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: row["category"])
    return {"has_secret_scanning_requirements": bool(rows), "requirements": rows}


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
