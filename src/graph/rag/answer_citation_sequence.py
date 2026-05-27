"""Audit numeric citation ordering in RAG answers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_CITATION_RE = re.compile(r"\[([0-9][0-9,\s]*)\]")


def audit_answer_citation_sequence(answer: str) -> dict[str, Any]:
    """Return deterministic citation sequence diagnostics for bracketed numbers."""
    citation_sequence = _citation_sequence(answer)
    first_seen_order = list(dict.fromkeys(citation_sequence))
    repeated_citations = [citation for citation, count in sorted(Counter(citation_sequence).items()) if count > 1]
    expected_order = sorted(first_seen_order)
    out_of_order_citations = [citation for citation, expected in zip(first_seen_order, expected_order, strict=True) if citation != expected]
    is_sequential = first_seen_order == list(range(1, max(first_seen_order, default=0) + 1)) and not out_of_order_citations
    return {
        "citation_sequence": citation_sequence,
        "first_seen_order": first_seen_order,
        "out_of_order_citations": out_of_order_citations,
        "repeated_citations": repeated_citations,
        "citation_count": len(citation_sequence),
        "is_sequential": is_sequential,
    }


def _citation_sequence(answer: str) -> list[int]:
    sequence: list[int] = []
    for match in _CITATION_RE.finditer(str(answer or "")):
        sequence.extend(int(part) for part in re.findall(r"\d+", match.group(1)))
    return sequence
