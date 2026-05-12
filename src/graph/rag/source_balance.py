"""Score source-project balance for RAG retrieval result sets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            return unit_metadata.get(key, _MISSING)
    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _source(result: Any) -> str:
    return _string(_value(result, "source_project")) or "unknown"


def score_source_balance(results: Iterable[Any], *, ideal_min_sources: int = 3) -> dict[str, Any]:
    """Summarize source concentration and actionable balance warnings."""
    ideal = _validate_positive_int(ideal_min_sources, "ideal_min_sources")
    counts = Counter(_source(result) for result in results)
    total = sum(counts.values())
    source_counts = dict(sorted(counts.items()))
    if not total:
        return {
            "total_results": 0,
            "source_count": 0,
            "dominant_source": None,
            "dominant_ratio": 0,
            "balance_score": 0,
            "source_counts": {},
            "warnings": ["no_results"],
        }

    dominant_source, dominant_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    dominant_ratio = dominant_count / total
    source_ratio = min(len(counts) / ideal, 1.0)
    concentration_penalty = max(0.0, (dominant_ratio - (1 / max(len(counts), 1))) / max(1 - (1 / max(len(counts), 1)), 1))
    balance_score = round(max(0.0, source_ratio * (1 - concentration_penalty)), 6)
    warnings = []
    if len(counts) < ideal:
        warnings.append("too_few_sources")
    if dominant_ratio >= 0.6 and total > 1:
        warnings.append("dominant_source_concentration")
    if "unknown" in counts:
        warnings.append("unknown_source_project")

    return {
        "total_results": total,
        "source_count": len(counts),
        "dominant_source": dominant_source,
        "dominant_ratio": round(dominant_ratio, 6),
        "balance_score": balance_score,
        "source_counts": source_counts,
        "warnings": warnings,
    }
