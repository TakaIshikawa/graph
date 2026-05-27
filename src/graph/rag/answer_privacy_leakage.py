"""Audit generated RAG answers for possible private data leakage."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.I | re.S,
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
_API_KEY_RE = re.compile(r"\b(?:sk|pk|api|key|token|ghp|github_pat|xox[baprs])[-_][A-Za-z0-9_=-]{16,}\b", re.I)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|bearer[_-]?token|client[_-]?secret|password|secret)"
    r"\s*[:=]\s*['\"]?([A-Za-z0-9_./+=-]{12,})['\"]?",
    re.I,
)
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z][A-Za-z0-9.-]+(?:\s+[A-Z][A-Za-z0-9.-]+){0,5}\s+"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court|Way|Pl|Place)\b"
)
_SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

_FINDERS: tuple[tuple[str, re.Pattern[str], Callable[[re.Match[str]], str], Callable[[str], bool]], ...] = (
    ("private_key", _PRIVATE_KEY_BLOCK_RE, lambda match: match.group(0), lambda _raw: True),
    ("email", _EMAIL_RE, lambda match: match.group(0), lambda _raw: True),
    ("phone", _PHONE_RE, lambda match: match.group(0), lambda _raw: True),
    ("api_key", _API_KEY_RE, lambda match: match.group(0), lambda _raw: True),
    ("secret_assignment", _SECRET_ASSIGNMENT_RE, lambda match: match.group(1), lambda _raw: True),
    ("physical_address", _ADDRESS_RE, lambda match: match.group(0), lambda _raw: True),
    ("ssn", _SSN_RE, lambda match: match.group(0), lambda _raw: True),
    ("credit_card", _CREDIT_CARD_RE, lambda match: match.group(0), lambda raw: _valid_luhn(raw)),
)


def audit_answer_privacy_leakage(answer: Any) -> dict[str, Any]:
    """Return possible privacy leakage categories with redacted samples only."""
    text = "" if answer is None else str(answer)
    samples = []
    risk_types: list[str] = []
    seen: set[tuple[str, str]] = set()
    line_starts = _line_starts(text)

    for risk_type, pattern, raw_value, is_valid in _FINDERS:
        for match in pattern.finditer(text):
            raw = raw_value(match)
            if not is_valid(raw):
                continue
            key = (risk_type, raw)
            if key in seen:
                continue
            seen.add(key)
            if risk_type not in risk_types:
                risk_types.append(risk_type)
            samples.append(
                {
                    "line_number": _line_number(line_starts, match.start()),
                    "risk_type": risk_type,
                    "redacted_value": _redact(raw, risk_type),
                }
            )

    return {
        "has_privacy_leakage_risk": bool(samples),
        "risk_types": risk_types,
        "samples": samples,
    }


def _line_starts(text: str) -> list[int]:
    return [0, *[match.end() for match in re.finditer(r"\n", text)]]


def _line_number(line_starts: list[int], position: int) -> int:
    line_number = 1
    for start in line_starts:
        if start > position:
            break
        line_number += 1
    return line_number - 1


def _redact(value: str, risk_type: str) -> str:
    compact = " ".join(value.split())
    if risk_type == "email":
        local, _, domain = compact.partition("@")
        return f"{_edge_mask(local)}@{_edge_mask(domain)}"
    if risk_type in {"credit_card", "ssn", "phone"}:
        digits = re.sub(r"\D", "", compact)
        return f"***{digits[-4:]}" if len(digits) >= 4 else "***"
    if risk_type == "private_key":
        return "[REDACTED PRIVATE KEY]"
    return _edge_mask(compact)


def _edge_mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}...{value[-2:]}"


def _valid_luhn(value: str) -> bool:
    digits = [int(char) for char in re.sub(r"\D", "", value)]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0
