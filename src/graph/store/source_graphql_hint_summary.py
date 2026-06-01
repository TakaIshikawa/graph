"""Summarize GraphQL hints on source records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_FIELD_KEYS = ("url", "endpoint", "api_url", "source_url", "type", "source_type", "description", "notes", "operation", "query")
_GRAPHQL_RE = re.compile(r"\bgraph[_\s-]?ql\b|/graphql\b", re.I)
_INTROSPECTION_RE = re.compile(r"\b(?:introspection|__schema|__type)\b", re.I)
_PERSISTED_RE = re.compile(r"\b(?:persisted\s+query|persisted_query|sha256hash|operation\s+hash)\b", re.I)
_OPERATIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("query", re.compile(r"\bquery\b", re.I)),
    ("mutation", re.compile(r"\bmutation\b", re.I)),
    ("subscription", re.compile(r"\bsubscription\b", re.I)),
)


def summarize_source_graphql_hints(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = graphql_sources = introspection = persisted = 0
    operation_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []

    for source in sources:
        total += 1
        hints = _hints(source)
        graphql_sources += bool(hints)
        introspection += any(kind == "introspection" for _, kind, _ in hints)
        persisted += any(kind == "persisted_query" for _, kind, _ in hints)
        for field, kind, value in hints:
            if kind.startswith("operation:"):
                operation_counts[kind.split(":", 1)[1]] += 1
            if len(samples) < limit:
                samples.append({"source_id": source_id(source), "field": field, "hint_type": kind, "value": value})

    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["field"]), sort_key(row["hint_type"])))
    return {
        "total_sources": total,
        "graphql_source_count": graphql_sources,
        "operation_counts": {key: operation_counts[key] for key in sorted(operation_counts, key=sort_key)},
        "introspection_hint_count": introspection,
        "persisted_query_count": persisted,
        "samples": samples[:limit],
    }


def _hints(source: Any) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field, value in _values(source):
        text = field_value(value)
        if not text:
            continue
        kinds = []
        if _GRAPHQL_RE.search(text):
            kinds.append("graphql")
        if _INTROSPECTION_RE.search(text):
            kinds.append("introspection")
        if _PERSISTED_RE.search(text):
            kinds.append("persisted_query")
        kinds.extend(f"operation:{name}" for name, pattern in _OPERATIONS if pattern.search(text))
        for kind in kinds:
            key = (field, kind)
            if key not in seen:
                rows.append((field, kind, text))
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
