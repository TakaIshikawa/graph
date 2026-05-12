"""Summarize evidence composition across RAG/search results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

_MISSING = object()
_SOURCE_TYPE_KEYS = ("source_type", "type", "kind")
_RELATION_KEYS = ("relation_type", "relation", "edge_type")
_CONFIDENCE_KEYS = ("confidence", "score", "source_confidence")
_DATE_KEYS = ("updated_at", "published_at", "created_at", "date", "source_date")


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING and value is not None:
            return value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING and value is not None:
            return value
        metadata = _field_value(unit, "metadata")
        if isinstance(metadata, Mapping):
            return metadata.get(key, _MISSING)

    return value


def _string(value: Any, default: str = "unknown") -> str:
    if value is _MISSING or value is None:
        return default
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text.casefold().replace(" ", "_") if text else default


def _first_string(result: Any, keys: tuple[str, ...], default: str = "unknown") -> str:
    for key in keys:
        value = _string(_result_value(result, key), default="")
        if value:
            return value
    return default


def _numeric(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _confidence_bucket(result: Any) -> str:
    value = None
    for key in _CONFIDENCE_KEYS:
        value = _numeric(_result_value(result, key))
        if value is not None:
            break
    if value is None:
        return "unknown"
    if value > 1:
        value = value / 100
    if value >= 0.75:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"


def _parse_date(value: Any) -> date | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return date.fromisoformat(text)
            except ValueError:
                return None
    return None


def _date_bucket(result: Any) -> str:
    for key in _DATE_KEYS:
        if _parse_date(_result_value(result, key)) is not None:
            return "dated"
    return "undated"


def _percentages(counter: Counter[str], total: int) -> dict[str, float]:
    if total == 0:
        return {}
    return {key: round((count / total) * 100, 1) for key, count in sorted(counter.items())}


def _counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _imbalances(total: int, counters: dict[str, Counter[str]]) -> list[str]:
    if total == 0:
        return ["no evidence results"]

    messages = []
    source_counts = counters["source_types"]
    if len(source_counts) <= 1:
        messages.append("missing source type diversity")
    for label, counter in sorted(counters.items()):
        if counter.get("unknown", 0):
            messages.append(f"missing {label.replace('_', ' ')} metadata")
        for key, count in sorted(counter.items()):
            if count / total >= 0.75 and total > 1:
                messages.append(f"dominant {label.replace('_', ' ')}: {key}")
    if counters["date_coverage"].get("undated", 0) == total:
        messages.append("missing date coverage")
    return messages


def analyze_result_evidence_mix(results: Iterable[Any]) -> dict[str, Any]:
    """Return counts, percentages, and imbalance notes for result evidence mix."""
    result_list = list(results)
    total = len(result_list)
    counters = {
        "source_types": Counter(),
        "confidence_buckets": Counter(),
        "relation_types": Counter(),
        "date_coverage": Counter(),
    }

    for result in result_list:
        counters["source_types"][_first_string(result, _SOURCE_TYPE_KEYS)] += 1
        counters["confidence_buckets"][_confidence_bucket(result)] += 1
        counters["relation_types"][_first_string(result, _RELATION_KEYS)] += 1
        counters["date_coverage"][_date_bucket(result)] += 1

    return {
        "total_results": total,
        "counts": {name: _counts(counter) for name, counter in counters.items()},
        "percentages": {
            name: _percentages(counter, total) for name, counter in counters.items()
        },
        "imbalances": _imbalances(total, counters),
    }
