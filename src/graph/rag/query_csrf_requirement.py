"""Detect CSRF protection requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_WEB_CONTEXT_RE = re.compile(r"\b(?:csrf|cross[-\s]?site\s+request\s+forgery|browser|web\s+form|html\s+form|session|cookie|post|put|patch|delete|unsafe\s+methods?)\b", re.I)
_STATE_ONLY_RE = re.compile(r"\b(?:state\s+management|application\s+state|state\s+machine)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("csrf_token", "high", (r"\bcsrf\s+tokens?\b", r"\banti[-\s]?csrf\s+tokens?\b", r"\brequest\s+forgery\s+tokens?\b")),
    ("double_submit_cookie", "medium", (r"\bdouble[-\s]?submit\s+cookies?\b", r"\bdouble\s+submit\s+csrf\b")),
    ("origin_check", "high", (r"\borigin\s+checks?\b", r"\breferer\s+checks?\b", r"\bvalidate\s+(?:origin|referer)\b")),
    ("same_site_cookie", "high", (r"\bsame\s*site\s+cookies?\b", r"\bsamesite\b")),
    ("state_parameter", "medium", (r"\boauth\s+state\s+parameter\b", r"\bstate\s+parameter\b", r"\bcsrf\s+state\b")),
    ("unsafe_methods", "high", (r"\bunsafe\s+(?:http\s+)?methods?\b", r"\b(?:post|put|patch|delete)\s+requests?\s+(?:need|require|must|should)\s+(?:csrf|forgery)\b", r"\b(?:post|put|patch|delete)\s+requests?\s+with\s+(?:csrf|forgery)\b")),
)


def detect_query_csrf_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    has_context = bool(_WEB_CONTEXT_RE.search(text)) and not (_STATE_ONLY_RE.search(text) and not re.search(r"\b(?:csrf|oauth|cookie|form|browser|session)\b", text, re.I))
    requirements = []
    if has_context:
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_csrf_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
