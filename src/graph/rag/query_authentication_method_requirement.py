"""Detect authentication method requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("passwordless_passkey", re.compile(r"\b(?:passwordless|passkeys?|webauthn|fido2)\b", re.I)),
    ("mfa", re.compile(r"\b(?:mfa|2fa|two[-\s]?factor|multi[-\s]?factor)\b", re.I)),
    ("sso", re.compile(r"\b(?:sso|single\s+sign[-\s]?on)\b", re.I)),
    ("oauth_oidc", re.compile(r"\b(?:oauth|openid\s+connect|oidc)\b", re.I)),
    ("saml", re.compile(r"\bsaml\b", re.I)),
    ("api_key", re.compile(r"\bapi\s+keys?\b", re.I)),
    ("service_account", re.compile(r"\bservice\s+accounts?\b", re.I)),
)


def detect_query_authentication_method_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, pattern in _SPECS:
        for match in pattern.finditer(text):
            rows.append({"category": category, "matched_text": match.group(0), "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["category"]))
    return rows
