"""Analyze license and rights signals in RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, string, value

_FIELDS = ("license", "rights", "copyright", "usage_rights", "terms")
_PERMISSIVE = ("creative commons", "cc-by", "cc by", "public domain", "open license", "mit", "apache")
_RESTRICTIVE = ("all rights reserved", "copyright", "proprietary", "no reuse", "restricted")


def analyze_result_license_signals(results: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    rows = list(results or [])
    counts: Counter[str] = Counter()
    samples = []
    permissive = restrictive = unknown = 0
    for index, result in enumerate(rows):
        license_text = _license_text(result)
        normalized = _normalize_license(license_text)
        counts[normalized] += 1
        category = _category(normalized)
        permissive += category == "permissive"
        restrictive += category == "restrictive"
        unknown += category == "unknown"
        if len(samples) < sample_limit:
            samples.append({"result_id": result_id(result, index), "title": string(value(result, "title")) or "", "license": normalized})
    return {
        "license_counts": dict(sorted(counts.items())),
        "permissive_count": permissive,
        "restrictive_count": restrictive,
        "unknown_count": unknown,
        "samples": samples,
    }


def _license_text(result: Any) -> str | None:
    for key in _FIELDS:
        text = string(value(result, key))
        if text:
            return text
    return None


def _normalize_license(text: str | None) -> str:
    if not text:
        return "unknown"
    lowered = text.casefold()
    if "public domain" in lowered or "cc0" in lowered:
        return "public_domain"
    if "creative commons" in lowered or "cc-by" in lowered or "cc by" in lowered:
        return "creative_commons"
    if "all rights reserved" in lowered:
        return "all_rights_reserved"
    for term in ("mit", "apache"):
        if term in lowered:
            return term
    return "_".join(lowered.split())


def _category(normalized: str) -> str:
    text = normalized.replace("_", " ")
    if any(term in text for term in _PERMISSIVE) or normalized in {"creative_commons", "public_domain", "mit", "apache"}:
        return "permissive"
    if any(term in text for term in _RESTRICTIVE) or normalized == "all_rights_reserved":
        return "restrictive"
    return "unknown"
