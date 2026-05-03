"""Analyze citation metadata coverage for RAG/search results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()

_URL_KEYS = (
    "url",
    "source_url",
    "canonical_url",
    "external_url",
    "link",
    "permalink",
    "uri",
)
_CITATION_KEYS = (
    "citation",
    "citation_url",
    "citation_count",
    "citations",
    "reference",
    "reference_url",
    "reference_count",
    "references",
    "source_citation",
    "source_citations",
)
_IDENTIFIER_KEYS = (
    "doi",
    "arxiv",
    "arxiv_id",
    "isbn",
    "isbn10",
    "isbn13",
    "pmid",
)
_SOURCE_KEYS = ("source", "source_name", "source_project", "source_id", "domain")


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _result_value(result: Any, key: str) -> Any:
    payload = _result_payload(result)
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


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _has_evidence(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value > 0
    if isinstance(value, Mapping):
        return any(_has_evidence(nested_value) for nested_value in value.values())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return any(_has_evidence(nested_value) for nested_value in value)
    return _string_value(value) is not None


def _iter_extra_keys(extras: Iterable[str] | str | None) -> Iterable[str]:
    if extras is None:
        return ()
    if isinstance(extras, str):
        return (extras,)
    return extras


def _normalize_keys(
    defaults: tuple[str, ...],
    extras: Iterable[str] | str | None,
) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for key in [*defaults, *_iter_extra_keys(extras)]:
        text = _string_value(key)
        if text is None or text in seen:
            continue
        seen.add(text)
        keys.append(text)
    return tuple(keys)


def _matching_keys(result: Any, keys: Iterable[str]) -> list[str]:
    return [key for key in keys if _has_evidence(_result_value(result, key))]


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _result_title(result: Any) -> str | None:
    return _string_value(_result_value(result, "title"))


def _result_source(result: Any) -> str | None:
    for key in _SOURCE_KEYS:
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return None


def _ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _missing_sort_key(item: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(item.get("source") or "").casefold(),
        str(item.get("title") or "").casefold(),
        str(item.get("id") or "").casefold(),
        int(item["index"]),
    )


def analyze_citation_coverage(
    results: Iterable[Any],
    *,
    citation_keys: Iterable[str] | str | None = None,
    url_keys: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """Return citation coverage counts, ratios, and per-result evidence flags."""
    url_key_set = _normalize_keys(_URL_KEYS, url_keys)
    citation_key_set = _normalize_keys(_CITATION_KEYS, citation_keys)
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for index, result in enumerate(list(results)):
        matched_url_keys = _matching_keys(result, url_key_set)
        matched_identifier_keys = _matching_keys(result, _IDENTIFIER_KEYS)
        matched_citation_keys = _matching_keys(result, citation_key_set)
        has_url = bool(matched_url_keys)
        has_identifier = bool(matched_identifier_keys)
        has_explicit_citation = bool(matched_citation_keys)
        has_citation = has_url or has_identifier or has_explicit_citation

        unit_id = _result_id(result, index)
        title = _result_title(result)
        source = _result_source(result)
        row = {
            "index": index,
            "id": unit_id,
            "title": title,
            "source": source,
            "has_url": has_url,
            "has_identifier": has_identifier,
            "has_explicit_citation": has_explicit_citation,
            "has_citation": has_citation,
            "url_keys": matched_url_keys,
            "identifier_keys": matched_identifier_keys,
            "citation_keys": matched_citation_keys,
        }
        rows.append(row)
        if not has_citation:
            missing.append(
                {
                    "index": index,
                    "id": unit_id,
                    "title": title,
                    "source": source,
                }
            )

    total = len(rows)
    with_url = sum(1 for row in rows if row["has_url"])
    with_identifier = sum(1 for row in rows if row["has_identifier"])
    with_explicit_citation = sum(1 for row in rows if row["has_explicit_citation"])
    with_citation = sum(1 for row in rows if row["has_citation"])
    missing_count = total - with_citation

    return {
        "total_results": total,
        "with_citation_count": with_citation,
        "with_url_count": with_url,
        "with_identifier_count": with_identifier,
        "with_explicit_citation_count": with_explicit_citation,
        "missing_citation_count": missing_count,
        "citation_coverage_ratio": _ratio(with_citation, total),
        "url_coverage_ratio": _ratio(with_url, total),
        "identifier_coverage_ratio": _ratio(with_identifier, total),
        "explicit_citation_coverage_ratio": _ratio(with_explicit_citation, total),
        "missing_citation_ratio": _ratio(missing_count, total),
        "results": rows,
        "missing_citations": sorted(missing, key=_missing_sort_key),
    }
