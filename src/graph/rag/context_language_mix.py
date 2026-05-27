"""Summarize language metadata for RAG context items."""

from __future__ import annotations

from typing import Any

from graph.rag._analysis_utils import string, value


def analyze_context_language_mix(context_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Return language counts and mix status for context items."""
    total = len(context_items or [])
    counts: dict[str, int] = {}
    missing = 0
    for item in context_items or []:
        lang = _language(item)
        if lang is None:
            missing += 1
        else:
            counts[lang] = counts.get(lang, 0) + 1
    dominant = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else None
    return {
        "total_items": total,
        "language_counts": dict(sorted(counts.items())),
        "missing_language_count": missing,
        "dominant_language": dominant,
        "mixed_language": len(counts) > 1,
    }


def _language(item: Any) -> str | None:
    for key in ("language", "lang"):
        text = string(value(item, key))
        if text:
            return text.casefold()
    return None
