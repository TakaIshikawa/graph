"""Summarize paywall and login-wall risk in evidence items."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, result_id, string, value

_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("subscription_required", re.compile(r"\b(?:subscription required|subscribe to|subscriber-only|subscribers only|paywall)\b", re.I)),
    ("sign_in_required", re.compile(r"\b(?:sign in|log in|login required|account required|registration required)\b", re.I)),
    ("abstract_only", re.compile(r"\b(?:abstract only|abstract available|no full text)\b", re.I)),
    ("purchase_access", re.compile(r"\b(?:purchase access|buy article|rent article|paid access)\b", re.I)),
)


def summarize_evidence_paywall_risk(evidence_items: Iterable[Any]) -> dict[str, Any]:
    items = list(evidence_items or [])
    cue_counts: Counter[str] = Counter()
    risk_items = []

    for index, item in enumerate(items):
        text = _item_text(item)
        cues = [label for label, pattern in _CUES if pattern.search(text)]
        if not cues:
            continue
        cue_counts.update(cues)
        risk_items.append({"id": result_id(item, index), "cues": cues})

    risk_count = len(risk_items)
    return {
        "total_items": len(items),
        "paywall_risk_count": risk_count,
        "risk_items": risk_items,
        "cue_counts": [{"cue": cue, "count": count} for cue, count in sorted(cue_counts.items())],
        "risk_ratio": round(risk_count / len(items), 4) if items else 0.0,
    }


def _item_text(item: Any) -> str:
    parts = [
        content_text(item),
        string(value(item, "access")) or "",
        string(value(item, "url")) or "",
        " ".join(iter_strings(metadata(item))),
        string(item) if isinstance(item, str) else "",
    ]
    return " ".join(part for part in parts if part).casefold()
