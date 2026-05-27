"""Audit quotation balance in generated answers."""

from __future__ import annotations

import re
from typing import Any

INLINE_QUOTE_RE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
WORD_RE = re.compile(r"\b[\w'-]+\b")
LONG_QUOTE_WORDS = 25
QUOTE_HEAVY_RATIO = 0.5


def audit_answer_quotation_balance(answer: object) -> dict[str, Any]:
    text = "" if answer is None else str(answer)
    quoted_segments = _quoted_segments(text)
    quoted_word_count = sum(_word_count(segment) for segment in quoted_segments)
    total_word_count = _word_count(text)
    ratio = round(quoted_word_count / total_word_count, 2) if total_word_count else 0.0
    flags: list[str] = []
    long_quote_count = sum(1 for segment in quoted_segments if _word_count(segment) > LONG_QUOTE_WORDS)
    if long_quote_count:
        flags.append("long_quote")
    if ratio > QUOTE_HEAVY_RATIO:
        flags.append("quote_heavy")
    if not quoted_segments:
        flags.append("no_quotes")
    return {
        "quote_count": len(quoted_segments),
        "quoted_word_count": quoted_word_count,
        "total_word_count": total_word_count,
        "quote_word_ratio": ratio,
        "long_quote_count": long_quote_count,
        "balance_flags": flags,
    }


def _quoted_segments(text: str) -> list[str]:
    segments: list[str] = []
    non_blockquote_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(">"):
            segments.append(stripped.lstrip("> "))
        else:
            non_blockquote_lines.append(line)
    remainder = "\n".join(non_blockquote_lines)
    for match in INLINE_QUOTE_RE.finditer(remainder):
        segments.append(next(group for group in match.groups() if group is not None))
    return [segment for segment in segments if segment.strip()]


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))
