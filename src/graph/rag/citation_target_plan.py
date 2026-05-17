"""Build deterministic citation target recommendations for RAG results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_ID_KEYS = ("id", "result_id", "unit_id", "source_id")
_SOURCE_KEYS = ("source", "source_name", "publisher", "domain", "source_project")
_URL_KEYS = ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri", "doi")
_EVIDENCE_KEYS = ("evidence", "evidence_items", "snippets", "quotes", "citations", "references")
_DATE_KEYS = ("published_at", "publication_date", "updated_at", "created_at", "date", "timestamp")
_SCORE_KEYS = ("score", "relevance_score", "rank_score", "density_score")


def build_citation_target_plan(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Group result-like objects into ranked citation targets."""
    groups: dict[str, dict[str, Any]] = {}
    labels: dict[str, str] = {}

    for index, result in enumerate(results):
        source_key = _source_key(result)
        group_key = source_key.casefold()
        labels.setdefault(group_key, source_key)
        group = groups.setdefault(
            group_key,
            {
                "result_ids": [],
                "evidence_count": 0,
                "best_score": None,
                "_latest_date": "",
            },
        )
        group["result_ids"].append(_result_id(result, index))
        group["evidence_count"] += _evidence_count(result)
        score = _score(result)
        if score is not None:
            group["best_score"] = score if group["best_score"] is None else max(group["best_score"], score)
        group["_latest_date"] = max(group["_latest_date"], _latest_date(result))

    rows = []
    for group_key, group in groups.items():
        best_score = group["best_score"]
        rows.append(
            {
                "source_key": labels[group_key],
                "result_ids": sorted(set(group["result_ids"])),
                "evidence_count": group["evidence_count"],
                "best_score": round(best_score, 3) if best_score is not None else None,
                "recommendation_reason": _reason(group["evidence_count"], group["_latest_date"], best_score),
                "_latest_date": group["_latest_date"],
            }
        )

    rows.sort(key=lambda item: str(item["source_key"]).casefold())
    rows.sort(key=lambda item: float(item["best_score"]) if item["best_score"] is not None else -1.0, reverse=True)
    rows.sort(key=lambda item: str(item["_latest_date"]), reverse=True)
    rows.sort(key=lambda item: int(item["evidence_count"]), reverse=True)
    for row in rows:
        row.pop("_latest_date", None)
    return rows


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
            metadata_value = unit_metadata.get(key, _MISSING)
            if metadata_value is not _MISSING and metadata_value is not None:
                return metadata_value
    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _has_value(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_value(item) for item in value.values())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return any(_has_value(item) for item in value)
    return True


def _count(value: Any) -> int:
    if not _has_value(value):
        return 0
    if isinstance(value, Mapping):
        return sum(1 for item in value.values() if _has_value(item))
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return sum(1 for item in value if _has_value(item))
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(int(value), 1)
    return 1


def _first_text(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        text = _string(_value(result, key))
        if text is not None:
            return text
    return None


def _source_key(result: Any) -> str:
    url = _first_text(result, _URL_KEYS)
    if url is not None:
        return url
    source = _first_text(result, _SOURCE_KEYS)
    title = _first_text(result, ("title",))
    if source and title:
        return f"{source}: {title}"
    if source:
        return source
    if title:
        return title
    return "unknown-source"


def _result_id(result: Any, index: int) -> str:
    text = _first_text(result, _ID_KEYS)
    return text if text is not None else f"result-{index + 1}"


def _evidence_count(result: Any) -> int:
    total = sum(_count(_value(result, key)) for key in _EVIDENCE_KEYS)
    return total or int(_has_value(_value(result, "snippet")) or _has_value(_value(result, "content")))


def _score(result: Any) -> float | None:
    for key in _SCORE_KEYS:
        value = _value(result, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
        text = _string(value)
        if text is not None:
            try:
                return float(text)
            except ValueError:
                continue
    return None


def _latest_date(result: Any) -> str:
    dates = [_string(_value(result, key)) or "" for key in _DATE_KEYS]
    return max(dates)


def _reason(evidence_count: int, latest_date: str, best_score: float | None) -> str:
    parts = []
    if evidence_count:
        parts.append(f"{evidence_count} evidence signal{'s' if evidence_count != 1 else ''}")
    if latest_date:
        parts.append("recent metadata")
    if best_score is not None:
        parts.append("strong retrieval score")
    return ", ".join(parts) if parts else "limited citation metadata"
