"""Detect OAuth authorization code flow requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_OAUTH_CONTEXT_RE = re.compile(
    r"\b(?:oauth\s*2(?:\.0)?|oauth2|oauth|oidc|openid\s+connect|authorization\s+code\s+(?:flow|grant)|auth\s+code\s+(?:flow|grant)|code\s+flow)\b",
    re.I,
)
_CODE_FLOW_SIGNAL_RE = re.compile(
    r"\b(?:authorization\s+endpoint|token\s+endpoint|redirect\s+uris?|redirect_uri|callback\s+uris?|authorization\s+codes?|code\s+exchange|exchange\s+the\s+code|confidential\s+clients?|client\s+secrets?)\b",
    re.I,
)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "authorization_code_exchange",
        "high",
        (
            r"\bauthorization\s+code\s+exchange\b",
            r"\bexchange\s+(?:the\s+)?authorization\s+codes?\b",
            r"\bexchange\s+(?:the\s+)?codes?\s+for\s+(?:access\s+)?tokens?\b",
            r"\bcode\s+exchange\b",
        ),
    ),
    (
        "authorization_endpoint",
        "high",
        (
            r"\bauthorization\s+endpoints?\b",
            r"\bauthorize\s+endpoints?\b",
            r"\bauthorization\s+urls?\b",
            r"\bauthorize\s+urls?\b",
        ),
    ),
    (
        "client_secret",
        "high",
        (
            r"\bclient\s+secrets?\b",
            r"\bconfidential\s+clients?\b",
            r"\bclient\s+authentication\b",
        ),
    ),
    (
        "redirect_uri",
        "high",
        (
            r"\bredirect\s+uris?\b",
            r"\bredirect_uri\b",
            r"\bcallback\s+uris?\b",
            r"\bredirect\s+urls?\b",
            r"\bcallback\s+urls?\b",
        ),
    ),
    (
        "token_endpoint",
        "high",
        (
            r"\btoken\s+endpoints?\b",
            r"\btokens?\s+urls?\b",
            r"\boauth\s+tokens?\s+endpoints?\b",
        ),
    ),
)


def detect_query_oauth_authorization_code_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    has_context = bool(_OAUTH_CONTEXT_RE.search(text) and _CODE_FLOW_SIGNAL_RE.search(text))

    requirements = []
    if has_context:
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_oauth_authorization_code_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
