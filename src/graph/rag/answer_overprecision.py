"""Detect overly precise answer claims not visibly supported by evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text

_CLAIMS = (
    ("currency", re.compile(r"[$€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?")),
    ("percentage", re.compile(r"\b\d+(?:\.\d+)?%")),
    ("date", re.compile(r"\b(?:20\d{2}|19\d{2})-\d{2}-\d{2}\b")),
    ("ranking", re.compile(r"(?<!\w)(?:#\d+|\d+(?:st|nd|rd|th))\b", re.I)),
    ("number", re.compile(r"(?<![\w$€£.#-])\d{1,3}(?:,\d{3})*(?:\.\d+)(?!\w)")),
)


def detect_answer_overprecision(answer: str, evidence: Iterable[Any] = (), *, max_decimal_places: int = 2) -> dict[str, Any]:
    """Flag precise claims absent from evidence text."""
    if not isinstance(max_decimal_places, int) or isinstance(max_decimal_places, bool) or max_decimal_places < 0:
        raise ValueError("max_decimal_places must be a non-negative integer")
    answer_text = str(answer or "")
    evidence_text = " ".join(content_text(item) for item in evidence).casefold()
    claims = []
    seen_spans: list[tuple[int, int]] = []
    for precision_type, pattern in _CLAIMS:
        for match in pattern.finditer(answer_text):
            span = match.span()
            if any(_overlaps(span, existing) for existing in seen_spans):
                continue
            seen_spans.append(span)
            matched = match.group(0)
            if not _is_precise(precision_type, matched, max_decimal_places):
                continue
            supported = bool(evidence_text and matched.casefold() in evidence_text)
            if not supported:
                claims.append(
                    {
                        "precision_type": precision_type,
                        "matched_text": matched,
                        "support_status": "unsupported",
                        "evidence": _snippet(answer_text, match.start(), match.end()),
                    }
                )
    counts = {kind: sum(1 for claim in claims if claim["precision_type"] == kind) for kind, _ in _CLAIMS}
    return {
        "claim_count": len(claims),
        "unsupported_count": len(claims),
        "precision_type_counts": counts,
        "claims": claims,
        "warnings": ["unsupported_precise_claims"] if claims else [],
    }


def _is_precise(kind: str, text: str, max_decimal_places: int) -> bool:
    if kind in {"currency", "percentage", "number"}:
        if "." in text:
            decimals = len(text.rsplit(".", 1)[1].rstrip("%"))
            return decimals > max_decimal_places or decimals > 0
        return kind in {"currency", "percentage"}
    return True


def _snippet(text: str, start: int, end: int) -> str:
    return text[max(0, start - 60) : min(len(text), end + 60)].strip()


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
