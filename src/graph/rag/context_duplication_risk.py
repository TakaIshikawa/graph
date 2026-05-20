"""Detect repeated or near-duplicate context in RAG result sets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import MISSING, content_text, result_id, rounded_ratio, string, tokens, value


def analyze_context_duplication_risk(
    results: Iterable[Any],
    *,
    min_overlap_ratio: float = 0.6,
) -> dict[str, Any]:
    """Return duplicate groups using stable IDs and deterministic ordering."""
    if isinstance(min_overlap_ratio, bool) or not isinstance(min_overlap_ratio, int | float) or not 0 < float(min_overlap_ratio) <= 1:
        raise ValueError("min_overlap_ratio must be between 0 and 1")

    items = [_candidate(result, index) for index, result in enumerate(results)]
    groups: list[dict[str, Any]] = []
    assigned: set[str] = set()

    for item in items:
        if item["id"] in assigned:
            continue
        members = [item]
        reasons: set[str] = set()
        for other in items:
            if other["id"] == item["id"] or other["id"] in assigned:
                continue
            reason = _duplicate_reason(item, other, float(min_overlap_ratio))
            if reason:
                members.append(other)
                reasons.add(reason)
        if len(members) > 1:
            member_ids = sorted((member["id"] for member in members), key=_sort_key)
            assigned.update(member_ids)
            groups.append({"result_ids": member_ids, "reasons": sorted(reasons), "group_size": len(member_ids)})

    duplicate_ids = {result_id_ for group in groups for result_id_ in group["result_ids"]}
    total = len(items)
    warnings = []
    duplicate_ratio = rounded_ratio(len(duplicate_ids), total)
    if duplicate_ratio >= 0.3 and total > 1:
        warnings.append("excessive_repeated_context")

    return {
        "total_results": total,
        "unique_count": total - len(duplicate_ids) + len(groups),
        "duplicate_count": len(duplicate_ids),
        "duplicate_ratio": duplicate_ratio,
        "duplicate_groups": sorted(groups, key=lambda group: (_sort_key(group["result_ids"][0]), group["group_size"])),
        "warnings": warnings,
    }


def _candidate(result: Any, index: int) -> dict[str, Any]:
    return {
        "id": result_id(result, index),
        "url": _normalized(_first_value(result, ("url", "source_url", "canonical_url"))),
        "metadata_id": _normalized(_first_value(result, ("metadata_id", "source_id", "unit_id"))),
        "tokens": tokens(content_text(result), min_length=3),
    }


def _duplicate_reason(left: dict[str, Any], right: dict[str, Any], threshold: float) -> str | None:
    if left["url"] and left["url"] == right["url"]:
        return "url"
    if left["metadata_id"] and left["metadata_id"] == right["metadata_id"]:
        return "metadata_id"
    if left["tokens"] and right["tokens"]:
        overlap = len(left["tokens"] & right["tokens"]) / max(1, min(len(left["tokens"]), len(right["tokens"])))
        if overlap >= threshold:
            return "content_overlap"
    return None


def _normalized(value_: Any) -> str | None:
    text = string(value_)
    return None if text is None else text.casefold()


def _first_value(result: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        item = value(result, key)
        if item is not MISSING and item is not None and string(item) is not None:
            return item
    return MISSING


def _sort_key(value_: object) -> tuple[str, str]:
    text = "" if value_ is None else str(value_)
    return (text.casefold(), text)
