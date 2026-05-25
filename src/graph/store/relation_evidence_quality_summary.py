"""Summarize relation evidence quality metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_EVIDENCE_KEYS = ("evidence", "evidence_ids", "supporting_evidence")
_RELATION_TYPE_KEYS = ("relation_type", "type", "predicate")
_CONFIDENCE_KEYS = ("confidence", "score", "evidence_confidence")
_STRENGTH_KEYS = ("strength", "quality")


def summarize_relation_evidence_quality(relations: Iterable[Any]) -> dict[str, Any]:
    """Aggregate missing, weak, and strong evidence counts for relations."""

    total = missing = weak = strong = 0
    evidence_counts: list[int] = []
    confidence_values: list[float] = []
    by_type: Counter[str | None] = Counter()

    for relation in relations:
        total += 1
        metadata = _metadata(relation)
        relation_type = _string(_first(relation, metadata, _RELATION_TYPE_KEYS))
        by_type[relation_type] += 1
        evidence = _evidence(relation, metadata)
        evidence_counts.append(len(evidence))
        confidence = _confidence(relation, metadata)
        if confidence is not None:
            confidence_values.append(confidence)
        if not evidence:
            missing += 1
        elif _is_strong(evidence, confidence):
            strong += 1
        else:
            weak += 1

    return {
        "missing_evidence_count": missing,
        "weak_evidence_count": weak,
        "strong_evidence_count": strong,
        "average_evidence_count": sum(evidence_counts) / total if total else 0.0,
        "average_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        "counts_by_relation_type": [
            {"relation_type": relation_type, "count": count}
            for relation_type, count in sorted(by_type.items(), key=lambda item: item[0] or "")
        ],
    }


def _evidence(item: Any, metadata: Mapping[str, Any]) -> list[Any]:
    for key in _EVIDENCE_KEYS:
        value = _get(item, key)
        if value not in (None, ""):
            return _as_list(value)
        value = metadata.get(key)
        if value not in (None, ""):
            return _as_list(value)
    return []


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    return [value] if value not in (None, "") else []


def _confidence(item: Any, metadata: Mapping[str, Any]) -> float | None:
    value = _first(item, metadata, _CONFIDENCE_KEYS)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _is_strong(evidence: list[Any], confidence: float | None) -> bool:
    if any(isinstance(item, Mapping) and str(item.get("strength", "")).lower() == "strong" for item in evidence):
        return True
    if confidence is not None and confidence >= 0.8:
        return True
    return len(evidence) >= 2


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _first(item: Any, metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _get(item, key)
        if value not in (None, ""):
            return value
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
