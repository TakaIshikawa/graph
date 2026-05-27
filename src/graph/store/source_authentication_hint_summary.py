"""Summarize authentication hints on source records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_SECRET_RE = re.compile(r"(token|secret|password|passwd|api[_-]?key|authorization|cookie|bearer|basic)", re.IGNORECASE)


def summarize_source_authentication_hints(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = with_hints = 0
    hint_types: Counter[str] = Counter()
    severities: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    for source in sources:
        total += 1
        hints = _hints(source)
        if hints:
            with_hints += 1
        for field, hint_type, severity, value in hints:
            hint_types[hint_type] += 1
            severities[severity] += 1
            if len(samples) < limit:
                samples.append({"source_id": source_id(source), "field": field, "hint_type": hint_type, "severity": severity, "value": _redact(value)})
    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["field"])))
    return {
        "total_sources": total,
        "sources_with_auth_hints": with_hints,
        "hint_type_counts": {key: hint_types[key] for key in sorted(hint_types, key=sort_key)},
        "severity_counts": {key: severities[key] for key in sorted(severities, key=sort_key)},
        "samples": samples[:limit],
    }


def _hints(source: Any) -> list[tuple[str, str, str, str]]:
    values: list[tuple[str, Any]] = []
    if isinstance(source, Mapping):
        values.extend(_walk(source))
    else:
        values.extend((key, get(source, key)) for key in ("authorization", "headers", "api_key", "token", "cookie", "metadata"))
    values.extend((f"metadata.{key}", value) for key, value in metadata(source).items())
    rows = []
    seen: set[tuple[str, str]] = set()
    for field, value in values:
        text = field_value(value)
        haystack = f"{field} {text}"
        if not _SECRET_RE.search(haystack):
            continue
        hint_type = _hint_type(haystack)
        severity = "high" if text and _SECRET_RE.search(field) else "medium"
        key = (field, hint_type)
        if key not in seen:
            rows.append((field, hint_type, severity, text))
            seen.add(key)
    return rows


def _walk(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, item in value.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            rows.extend(_walk(item, field))
        else:
            rows.append((field, item))
    return rows


def _hint_type(text: str) -> str:
    lowered = text.casefold()
    if "bearer" in lowered:
        return "bearer"
    if "basic" in lowered:
        return "basic"
    if "cookie" in lowered:
        return "cookie"
    if "api_key" in lowered or "api-key" in lowered or "apikey" in lowered:
        return "api_key"
    return "token"


def _redact(value: str) -> str:
    return "" if not value else "[redacted]"
