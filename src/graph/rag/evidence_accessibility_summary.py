"""Summarize access status for RAG evidence records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, rounded_ratio, string, value

_BUCKETS = ("open", "restricted", "paywalled", "login_required", "missing_access", "unknown")


def summarize_evidence_accessibility(evidence: Iterable[Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    restricted_ids = []
    total = 0
    for index, item in enumerate(evidence or []):
        total += 1
        bucket = _bucket(item)
        counts[bucket] += 1
        if bucket in {"restricted", "paywalled", "login_required"}:
            restricted_ids.append(result_id(item, index))
    return {
        "total_items": total,
        "bucket_counts": {bucket: counts[bucket] for bucket in _BUCKETS},
        "restricted_ids": sorted(restricted_ids),
        "missing_access_count": counts["missing_access"],
        "open_ratio": round(rounded_ratio(counts["open"], total), 3),
    }


def _bucket(item: Any) -> str:
    if _truthy(value(item, "paywalled")):
        return "paywalled"
    if _truthy(value(item, "login_required")):
        return "login_required"
    for key in ("access", "access_status", "license"):
        text = string(value(item, key))
        if not text:
            continue
        lowered = text.casefold()
        if any(cue in lowered for cue in ("open", "public", "cc-by", "creative commons")):
            return "open"
        if "paywall" in lowered or "subscription" in lowered:
            return "paywalled"
        if "login" in lowered or "account" in lowered:
            return "login_required"
        if any(cue in lowered for cue in ("restricted", "private", "invite")):
            return "restricted"
        return "unknown"
    return "missing_access"


def _truthy(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    text = string(raw)
    return text is not None and text.casefold() in {"true", "yes", "1", "paywalled", "required"}
