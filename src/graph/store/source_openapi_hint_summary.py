"""Summarize OpenAPI and Swagger documentation hints on source records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_FIELD_KEYS = (
    "url",
    "uri",
    "endpoint",
    "api_url",
    "source_url",
    "documentation_url",
    "description",
    "notes",
    "title",
    "type",
    "source_type",
)
_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openapi", re.compile(r"\bopenapi(?:\.json|\.ya?ml)?\b|/openapi(?:\.json|\.ya?ml)?\b", re.I)),
    ("swagger", re.compile(r"\bswagger(?:\s+ui|\.json|\.ya?ml)?\b|/swagger(?:\.json|\.ya?ml|/ui)?\b", re.I)),
    ("api_docs", re.compile(r"\bapi[-_\s]?docs?\b|/api-docs\b|/api/docs\b", re.I)),
    ("redoc", re.compile(r"\bredoc(?:ly)?\b|/redoc\b", re.I)),
    ("scalar", re.compile(r"\bscalar\b|/scalar\b", re.I)),
)
_SPEC_URL_RE = re.compile(r"https?://[^\s\"'<>]+(?:openapi|swagger|api-docs)[^\s\"'<>]*", re.I)


def summarize_source_openapi_hints(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = hinted = 0
    hint_counts: Counter[str] = Counter()
    spec_urls: set[str] = set()
    samples: list[dict[str, str]] = []

    for source in sources:
        total += 1
        hints = _hints(source)
        hinted += bool(hints)
        for field, category, value in hints:
            hint_counts[category] += 1
            spec_urls.update(_likely_spec_urls(value))
            if len(samples) < limit:
                samples.append({"source_id": source_id(source), "field": field, "category": category, "matched_text": value})

    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["field"]), sort_key(row["category"])))
    return {
        "total_sources": total,
        "sources_with_openapi_hints": hinted,
        "hint_counts": {key: hint_counts[key] for key in sorted(hint_counts, key=sort_key)},
        "likely_spec_urls": sorted(spec_urls, key=sort_key),
        "samples": samples[:limit],
    }


def _hints(source: Any) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field, value in _values(source):
        text = field_value(value)
        if not text:
            continue
        for category, pattern in _HINTS:
            if not pattern.search(text):
                continue
            key = (field, category)
            if key not in seen:
                rows.append((field, category, text))
                seen.add(key)
    return rows


def _values(source: Any) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if isinstance(source, Mapping):
        values.extend(_walk(source))
    else:
        values.extend((key, get(source, key)) for key in _FIELD_KEYS)
    values.extend((f"metadata.{key}", value) for key, value in metadata(source).items())
    return values


def _walk(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, item in value.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            rows.extend(_walk(item, field))
        else:
            rows.append((field, item))
    return rows


def _likely_spec_urls(text: str) -> list[str]:
    return [match.group(0).rstrip(".,)") for match in _SPEC_URL_RE.finditer(text)]
