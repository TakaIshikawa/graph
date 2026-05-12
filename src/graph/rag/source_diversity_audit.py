"""Audit source diversity across retrieved RAG results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()
_URL_KEYS = ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri")
_FACETS = ("source_project", "source_entity_type", "content_type", "citation_host")


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_threshold(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("dominance_threshold must be between 0 and 1")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("dominance_threshold must be between 0 and 1") from exc
    if threshold <= 0 or threshold > 1:
        raise ValueError("dominance_threshold must be between 0 and 1")
    return threshold


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


def _id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _normalize_value(value: Any) -> str:
    return (_string(value) or "unknown").casefold()


def _host_from_url(value: Any) -> str | None:
    text = _string(value)
    if text is None:
        return None
    parsed = urlsplit(text if "://" in text else f"https://{text}")
    host = parsed.hostname or parsed.netloc
    if host is None:
        return None
    host = host.casefold().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _citation_host(result: Any) -> str:
    for key in _URL_KEYS:
        host = _host_from_url(_value(result, key))
        if host is not None:
            return host
    return "unknown"


def _facet_value(result: Any, facet: str) -> str:
    if facet == "citation_host":
        return _citation_host(result)
    return _normalize_value(_value(result, facet))


def audit_source_diversity(
    results: Iterable[Any],
    *,
    max_examples: int = 3,
    dominance_threshold: float = 0.6,
) -> dict[str, Any]:
    """Summarize source diversity by configured result facets."""
    max_examples_value = _validate_positive_int(max_examples, "max_examples")
    threshold = _validate_threshold(dominance_threshold)
    result_list = list(results)
    result_ids = [_id(result, index) for index, result in enumerate(result_list)]
    facet_values: dict[str, dict[str, list[str]]] = {
        facet: defaultdict(list) for facet in _FACETS
    }

    for result, result_id in zip(result_list, result_ids, strict=False):
        for facet in _FACETS:
            facet_values[facet][_facet_value(result, facet)].append(result_id)

    distributions: dict[str, list[dict[str, Any]]] = {}
    warnings: list[dict[str, Any]] = []
    total = len(result_list)
    for facet in _FACETS:
        rows = []
        for value, ids in facet_values[facet].items():
            ratio = round(len(ids) / total, 3) if total else 0.0
            rows.append(
                {
                    "value": value,
                    "count": len(ids),
                    "ratio": ratio,
                    "representative_result_ids": ids[:max_examples_value],
                }
            )
        rows.sort(key=lambda item: (-item["count"], item["value"]))
        distributions[facet] = rows
        for row in rows:
            if total and row["ratio"] > threshold:
                warnings.append(
                    {
                        "facet": facet,
                        "value": row["value"],
                        "ratio": row["ratio"],
                        "message": (
                            f"{facet} value {row['value']} represents "
                            f"{row['count']} of {total} results"
                        ),
                    }
                )

    warnings.sort(key=lambda item: (item["facet"], item["value"]))
    return {
        "totals": {
            "result_count": total,
            "facet_count": len(_FACETS),
            "dominance_threshold": threshold,
        },
        "distributions": distributions,
        "dominance_warnings": warnings,
    }
