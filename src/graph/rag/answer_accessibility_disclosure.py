"""Audit accessibility disclosure in RAG answers."""

from __future__ import annotations

import re
from typing import Any

_ACCESS_RE = re.compile(r"\baccessib|wcag|aria|screen\s+reader|alt\s+text|caption|keyboard|focus\s+order\b", re.I)
_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("WCAG", re.compile(r"\bwcag\b", re.I)),
    ("screen_readers", re.compile(r"\bscreen\s+readers?\b|voiceover|nvda|jaws", re.I)),
    ("keyboard", re.compile(r"\bkeyboard\b|focus\s+order", re.I)),
    ("captions", re.compile(r"\bcaption(?:s|ed)?\b|subtitles?", re.I)),
    ("alt_text", re.compile(r"\balt\s+text\b|alternative\s+text", re.I)),
    ("ARIA", re.compile(r"\baria\b|aria-[a-z0-9_-]+", re.I)),
)


def audit_answer_accessibility_disclosure(answer: str, query: str = "") -> dict[str, Any]:
    answer_text = str(answer or "")
    query_text = str(query or "")
    required = bool(_ACCESS_RE.search(query_text))
    signals = [name for name, pattern in _SIGNALS if pattern.search(answer_text)]
    return {
        "required": required,
        "disclosed": bool(signals),
        "signals": signals,
        "recommendation": "add_accessibility_considerations" if required and not signals else "",
    }
