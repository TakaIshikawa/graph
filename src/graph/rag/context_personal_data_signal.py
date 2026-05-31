"""Analyze retrieved RAG context for personal data signals."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, string

_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")),
    (
        "physical_address",
        re.compile(
            r"\b\d{1,6}\s+[A-Z][A-Za-z0-9.-]+(?:\s+[A-Z][A-Za-z0-9.-]+){0,5}\s+"
            r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Lane|Ln|Way|Court|Ct)\b"
        ),
    ),
    (
        "api_key_or_token",
        re.compile(
            r"\b(?:api[_\s-]?key|access[_\s-]?token|auth[_\s-]?token|bearer|secret|password)\s*[:=]\s*[A-Za-z0-9._~+/=-]{8,}\b",
            re.I,
        ),
    ),
    (
        "government_id_label",
        re.compile(r"\b(?:ssn|social security|passport(?: number)?|driver'?s license|tax id|national id)\b", re.I),
    ),
    ("personal_name_label", re.compile(r"\b(?:full name|legal name|patient name|customer name|employee name)\s*[:=]", re.I)),
)
_SIGNAL_NAMES = tuple(name for name, _ in _SIGNALS)


def analyze_context_personal_data_signals(context_items: Iterable[Any]) -> dict[str, Any]:
    """Return aggregate personal data signals without exposing raw secret values."""
    examples = []
    counts: Counter[str] = Counter()
    risky_indexes: set[int] = set()

    for index, item in enumerate(context_items):
        text = _context_text(item)
        if not text:
            continue
        item_signals = []
        for signal_type, pattern in _SIGNALS:
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            counts[signal_type] += len(matches)
            item_signals.append(signal_type)
        if item_signals:
            risky_indexes.add(index)
            for signal_type in item_signals:
                examples.append({"item_index": index, "signal_type": signal_type})

    total_signals = sum(counts.values())
    return {
        "signal_counts": {name: counts.get(name, 0) for name in _SIGNAL_NAMES},
        "risky_item_count": len(risky_indexes),
        "risk_level": _risk_level(total_signals, len(risky_indexes)),
        "examples": examples[:10],
    }


def _context_text(item: Any) -> str:
    direct = string(item) if isinstance(item, str | bytes) else None
    if direct is not None:
        return direct
    return content_text(item)


def _risk_level(total_signals: int, risky_item_count: int) -> str:
    if total_signals >= 4 or risky_item_count >= 3:
        return "high"
    if total_signals >= 2 or risky_item_count >= 2:
        return "medium"
    if total_signals:
        return "low"
    return "low"
