"""Classify context records by primary-source signals."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, domain_for, result_id, string, value

_PRIMARY_SOURCE_TYPES = ("primary", "official", "journal", "standard", "dataset", "regulation")
_SECONDARY_SOURCE_TYPES = ("blog", "news", "summary", "analysis", "review")
_OFFICIAL_SUFFIXES = (".gov", ".mil", ".edu", ".int")


def analyze_context_primary_source_signals(context_items: Iterable[Any]) -> dict[str, Any]:
    rows = []
    counts = {"primary_source": 0, "secondary_source": 0, "unknown": 0}
    for index, item in enumerate(context_items or []):
        classification = _classify(item)
        counts[classification] += 1
        rows.append({"context_id": result_id(item, index), "classification": classification, "title": string(value(item, "title")) or ""})
    return {"classification_counts": counts, "contexts": rows}


def _classify(item: Any) -> str:
    source_type = (string(value(item, "source_type")) or "").casefold()
    text = content_text(item).casefold()
    domain = domain_for(item) or ""
    if any(term in source_type for term in _PRIMARY_SOURCE_TYPES) or domain.endswith(_OFFICIAL_SUFFIXES) or "official" in text:
        return "primary_source"
    if any(term in source_type for term in _SECONDARY_SOURCE_TYPES) or any(term in text for term in ("blog", "news summary", "reported by")):
        return "secondary_source"
    return "unknown"
