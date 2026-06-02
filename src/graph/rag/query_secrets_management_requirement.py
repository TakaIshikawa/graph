"""Detect secrets management requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(
    r"\b(?:secrets?|credentials?|api\s+keys?|tokens?|vaults?|service\s+accounts?|environment\s+variables?|secret\s+scanning|least[-\s]privilege|security)\b",
    re.I,
)
_SENSITIVE_TOKEN_RE = re.compile(r"\b(?:api\s+keys?|tokens?)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("access_control", "high", (r"\bleast[-\s]privilege\b", r"\baccess\s+controls?\b", r"\biam\b", r"\bscoped\s+(?:access|permissions?)\b")),
    ("api_keys", "high", (r"\bapi\s+keys?\b",)),
    ("env_vars", "medium", (r"\benvironment\s+variables?\b", r"\benv\s+vars?\b", r"\b\.env\b")),
    ("rotation", "high", (r"\brotat(?:e|ion|ing)\b", r"\bexpir(?:e|es|ing|ation)\b")),
    ("scanning", "medium", (r"\bsecret\s+scanning\b", r"\bcredential\s+scanning\b", r"\bscan\s+(?:for\s+)?secrets?\b")),
    ("service_account", "high", (r"\bservice\s+accounts?\b", r"\bmachine\s+credentials?\b")),
    ("storage", "high", (r"\bsecret\s+storage\b", r"\bstor(?:e|age|ing)\s+(?:secrets?|credentials?|api\s+keys?|tokens?)\b")),
    ("tokens", "high", (r"\btokens?\b",)),
    ("vault", "high", (r"\bvaults?\b", r"\bkey\s+vault\b", r"\bsecrets?\s+manager\b")),
)


def detect_query_secrets_management_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    rows: list[dict[str, Any]] = []
    has_context = bool(_CONTEXT_RE.search(text))
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if not match:
            continue
        if category in {"api_keys", "tokens"} and not _credential_context(text, match):
            continue
        if has_context:
            rows.append({"category": category, "matched_text": match.group(0), "severity": severity})
    rows.sort(key=lambda row: row["category"])
    return {"has_secrets_management_requirement": bool(rows), "requirements": rows}


def _credential_context(text: str, match: re.Match[str]) -> bool:
    window = text[max(0, match.start() - 80) : match.end() + 80]
    return bool(
        re.search(
            r"\b(?:secrets?|credentials?|security|vaults?|rotat(?:e|ion|ing)|stor(?:e|age|ing)|scann(?:ing|er)|least[-\s]privilege|service\s+accounts?)\b",
            window,
            re.I,
        )
    )


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
