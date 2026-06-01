"""Analyze retrieved context for likely PII exposure."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._record_text import text_blob

_PII: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("address", re.compile(r"\b\d{1,6}\s+[A-Z][A-Za-z0-9.-]+(?:\s+[A-Z][A-Za-z0-9.-]+){0,4}\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Drive|Dr|Ln|Lane)\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
)


def analyze_context_pii_exposure_signals(context_items: Iterable[Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    pii_indexes: set[int] = set()
    for index, item in enumerate(context_items):
        text = text_blob(item)
        for name, pattern in _PII:
            matches = pattern.findall(text)
            if matches:
                counts[name] += len(matches)
                pii_indexes.add(index)
    type_counts = {name: counts.get(name, 0) for name, _ in _PII}
    return {
        "pii_item_count": len(pii_indexes),
        "pii_type_counts": type_counts,
        "redaction_recommended": any(type_counts.values()),
    }
