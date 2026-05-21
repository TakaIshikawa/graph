"""Classify retrieved RAG results by primary, secondary, tertiary, or unknown source type."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import domain_for, result_id, string, value

_PRIMARY_TYPES = {"primary", "official", "dataset", "filing", "transcript", "standard", "source_code", "report"}
_SECONDARY_TYPES = {"secondary", "article", "review", "analysis", "news", "commentary"}
_TERTIARY_TYPES = {"tertiary", "encyclopedia", "wiki", "glossary", "index", "directory"}
_TYPE_KEYS = ("source_type", "publication_type", "document_type")
_TEXT_KEYS = ("title", "publisher", "author", "url", "domain")
_PRIMARY_RE = re.compile(r"\b(official|original report|dataset|filing|10-k|transcript|standard|specification|source code|github)\b", re.I)
_SECONDARY_RE = re.compile(r"\b(analysis|review|news|commentary|explainer|blog)\b", re.I)
_TERTIARY_RE = re.compile(r"\b(wiki|encyclopedia|glossary|directory|index)\b", re.I)


def classify_result_primary_sources(results: Iterable[Any]) -> dict[str, Any]:
    """Classify source category using metadata-first deterministic rules."""
    rows = []
    counts: Counter[str] = Counter()
    for index, result in enumerate(results or []):
        category, reasons = _classify(result)
        counts[category] += 1
        rows.append({"result_id": result_id(result, index), "category": category, "reasons": reasons})

    reason_counts = Counter(reason for row in rows for reason in row["reasons"])
    category_counts = {category: counts.get(category, 0) for category in ("primary", "secondary", "tertiary", "unknown")}
    return {
        "total_results": len(rows),
        "primary_count": category_counts["primary"],
        "secondary_count": category_counts["secondary"],
        "tertiary_count": category_counts["tertiary"],
        "unknown_count": category_counts["unknown"],
        "category_counts": category_counts,
        "results": rows,
        "reason_counts": dict(sorted(reason_counts.items())),
        "warnings": ["no_results"] if not rows else (["unknown_source_category"] if category_counts["unknown"] else []),
    }


def _classify(result: Any) -> tuple[str, list[str]]:
    for key in _TYPE_KEYS:
        text = string(value(result, key))
        if not text:
            continue
        normalized = "_".join(text.casefold().split())
        if normalized in _PRIMARY_TYPES:
            return "primary", [f"metadata_{key}_primary"]
        if normalized in _SECONDARY_TYPES:
            return "secondary", [f"metadata_{key}_secondary"]
        if normalized in _TERTIARY_TYPES:
            return "tertiary", [f"metadata_{key}_tertiary"]

    haystack = " ".join(filter(None, [string(value(result, key)) for key in _TEXT_KEYS] + [domain_for(result)]))
    if _TERTIARY_RE.search(haystack):
        return "tertiary", ["tertiary_heuristic"]
    if _PRIMARY_RE.search(haystack):
        return "primary", ["primary_heuristic"]
    if _SECONDARY_RE.search(haystack):
        return "secondary", ["secondary_heuristic"]
    return "unknown", ["insufficient_source_signals"]
