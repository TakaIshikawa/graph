"""Summarize entity diversity in context items."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, value

_ENTITY_KEYS = ("entities", "people", "organizations", "authors")
_CAP_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3})\b")
_BLOCKLIST = {"The", "This", "That", "A", "An"}


def summarize_context_entity_diversity(context_items: Iterable[Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for item in context_items:
        for key in _ENTITY_KEYS:
            for entity in iter_strings(value(item, key)):
                counts[_normalize(entity)] += 1
        for match in _CAP_RE.findall(content_text(item)):
            if match not in _BLOCKLIST:
                counts[_normalize(match)] += 1
    total = sum(counts.values())
    repeated = sum(count for count in counts.values() if count > 1)
    top = [{"entity": entity, "count": count} for entity, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]
    return {
        "entity_counts": dict(sorted(counts.items())),
        "top_entities": top,
        "repeated_entity_ratio": repeated / total if total else 0.0,
        "singleton_count": sum(1 for count in counts.values() if count == 1),
    }


def _normalize(entity: str) -> str:
    return " ".join(str(entity).split())
