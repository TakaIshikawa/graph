"""Detect multi-factor authentication requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_AUTH_CONTEXT_RE = re.compile(r"\b(?:auth(?:entication)?|login|log\s*in|sign[-\s]?in|access|account|user|admin|privileged)\b", re.I)
_MFA_CONTEXT_RE = re.compile(r"\b(?:mfa|multi[-\s]?factor|two[-\s]?factor|2fa|one[-\s]?time\s+(?:password|code)|otp|step[-\s]?up)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("admin_enforcement", "high", (r"\b(?:admins?|administrators?|privileged\s+users?)\s+(?:must|should)\s+(?:require|enforce|use|enable)\s+(?:mfa|multi[-\s]?factor|two[-\s]?factor|2fa)\b", r"\b(?:admins?|administrators?|privileged\s+users?)\s+(?:require|enforce|use|enable)\s+(?:mfa|multi[-\s]?factor|two[-\s]?factor|2fa)\b", r"\b(?:enforce|require)\s+(?:mfa|multi[-\s]?factor|two[-\s]?factor|2fa)\s+for\s+(?:admins?|administrators?|privileged\s+users?)\b")),
    ("authenticator_app", "medium", (r"\bauthenticator\s+apps?\b", r"\btotp\b", r"\btime[-\s]?based\s+one[-\s]?time\s+password\b")),
    ("hardware_key", "high", (r"\bhardware\s+(?:security\s+)?keys?\b", r"\bsecurity\s+keys?\b", r"\bfido2\b", r"\bwebauthn\b", r"\byubikeys?\b")),
    ("recovery_codes", "medium", (r"\brecovery\s+codes?\b", r"\bbackup\s+codes?\b")),
    ("sms_fallback", "medium", (r"\bsms\s+(?:fallback|otp|code|verification)\b", r"\btext\s+message\s+(?:fallback|code|verification)\b")),
    ("step_up_authentication", "high", (r"\bstep[-\s]?up\s+auth(?:entication)?\b", r"\bre[-\s]?authenticate\s+for\s+(?:sensitive|high[-\s]?risk|privileged)\b", r"\badditional\s+factor\s+for\s+(?:sensitive|high[-\s]?risk|privileged)\b")),
)


def detect_query_mfa_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    has_context = bool(_AUTH_CONTEXT_RE.search(text) and _MFA_CONTEXT_RE.search(text))
    requirements = []
    if has_context:
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_mfa_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
