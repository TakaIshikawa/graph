"""Summarize WebSocket and realtime hints on source records."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_FIELD_KEYS = (
    "url",
    "uri",
    "endpoint",
    "api_url",
    "source_url",
    "type",
    "source_type",
    "protocol",
    "transport",
    "description",
    "notes",
)
_WEBSOCKET_RE = re.compile(r"\b(?:websocket|socket\.io)\b|wss?://", re.I)
_SECURE_RE = re.compile(r"\bwss://", re.I)
_REALTIME_RE = re.compile(r"\b(?:realtime|real-time|subscription|streaming\s+api)\b", re.I)


def summarize_source_websocket_hints(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = websocket_sources = secure_websocket = realtime_hints = 0
    samples: list[dict[str, str]] = []

    for source in sources:
        total += 1
        hints = _hints(source)
        has_websocket = any(kind in {"websocket", "secure_websocket"} for _, kind, _ in hints)
        has_secure = any(kind == "secure_websocket" for _, kind, _ in hints)
        has_realtime = any(kind == "realtime" for _, kind, _ in hints)
        websocket_sources += has_websocket
        secure_websocket += has_secure
        realtime_hints += has_realtime
        for field, kind, value in hints:
            if len(samples) < limit:
                samples.append({"source_id": source_id(source), "field": field, "hint_type": kind, "value": value})

    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["field"]), sort_key(row["hint_type"])))
    return {
        "total_sources": total,
        "websocket_source_count": websocket_sources,
        "secure_websocket_count": secure_websocket,
        "realtime_hint_count": realtime_hints,
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
        if _SECURE_RE.search(text):
            kinds.append("secure_websocket")
        elif _WEBSOCKET_RE.search(text):
            kinds.append("websocket")
        if _REALTIME_RE.search(text):
            kinds.append("realtime")
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
