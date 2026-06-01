"""Summarize URL-bearing fields in unit frontmatter metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_DEFAULT_FIELD_HINTS = ("url", "uri", "link", "canonical", "source")
_FRONTMATTER_KEYS = ("frontmatter", "frontmatter_metadata", "yaml_frontmatter")


def summarize_unit_frontmatter_url_fields(units: Iterable[Any], field_names: Iterable[str] | None = None, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    selected = {name.casefold() for name in field_names or []}
    url_field_counts: Counter[str] = Counter()
    scheme_counts: Counter[str] = Counter()
    field_samples: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    invalid_url_samples: list[dict[str, str]] = []
    missing_scheme_count = 0

    for unit in units:
        uid = unit_id(unit)
        for field, value in _frontmatter_items(unit):
            if selected and field.casefold() not in selected:
                continue
            if not selected and not _looks_like_url_field(field):
                continue
            text = field_value(value)
            if not text:
                continue
            url_field_counts[field] += 1
            parsed = urlparse(text)
            if parsed.scheme:
                if _valid_url(parsed):
                    scheme_counts[parsed.scheme.casefold()] += 1
                elif len(invalid_url_samples) < limit:
                    invalid_url_samples.append({"unit_id": uid, "field": field, "value": text})
            else:
                missing_scheme_count += 1
            if len(field_samples[field]) < limit:
                field_samples[field].append({"unit_id": uid, "value": text})

    return {
        "url_field_counts": _sorted_counts(url_field_counts),
        "scheme_counts": _sorted_counts(scheme_counts),
        "missing_scheme_count": missing_scheme_count,
        "invalid_url_samples": sorted(invalid_url_samples, key=lambda row: (sort_key(row["unit_id"]), sort_key(row["field"]), sort_key(row["value"]))),
        "field_samples": {field: field_samples[field] for field in sorted(field_samples, key=sort_key)},
    }


def _frontmatter_items(unit: Any) -> list[tuple[str, Any]]:
    meta = metadata(unit)
    rows: list[tuple[str, Any]] = []
    for key, value in meta.items():
        if isinstance(value, Mapping) and key in _FRONTMATTER_KEYS:
            rows.extend((str(nested_key), nested_value) for nested_key, nested_value in value.items() if _scalar(nested_value))
        elif _scalar(value):
            rows.append((str(key), value))
    for key in _FRONTMATTER_KEYS:
        value = get(unit, key)
        if isinstance(value, Mapping):
            rows.extend((str(nested_key), nested_value) for nested_key, nested_value in value.items() if _scalar(nested_value))
    return rows


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _looks_like_url_field(field: str) -> bool:
    lowered = field.casefold()
    return any(hint in lowered for hint in _DEFAULT_FIELD_HINTS)


def _valid_url(parsed: Any) -> bool:
    if parsed.scheme.casefold() in {"http", "https", "ftp"}:
        return bool(parsed.netloc)
    return bool(parsed.path or parsed.netloc)


def _sorted_counts(counts: Counter[str]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts, key=sort_key)}
