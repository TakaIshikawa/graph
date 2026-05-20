"""Extract version-like signals from retrieved RAG results."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import MISSING, result_id, string, value

_DEFAULT_TEXT_FIELDS = ("title", "content", "text", "summary", "snippet")
_METADATA_FIELDS = ("version", "api_version", "model", "model_name", "release", "release_label")
_PATTERNS = (
    ("semantic_version", re.compile(r"\b(?:v)?\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?\b")),
    ("api_version", re.compile(r"\bv\d+(?:\.\d+)?\b|\bapi[-_ ]?version[-_ ]?\d+(?:\.\d+)?\b", re.I)),
    ("model_name", re.compile(r"\b(?:gpt|claude|gemini|llama|mistral|o)[-_ ]?\d(?:[A-Za-z0-9._-]*)\b", re.I)),
    ("release_label", re.compile(r"\b(?:alpha|beta|rc\d*|preview|stable|lts|ga)\b", re.I)),
    ("date_version", re.compile(r"\b(?:20\d{2})[-.](?:0[1-9]|1[0-2])[-.](?:0[1-9]|[12]\d|3[01])\b")),
)


def extract_result_version_signals(results: Iterable[Any], *, text_fields: Iterable[str] | None = None) -> dict[str, Any]:
    """Return deduplicated per-result version signals and aggregate counts."""
    fields = tuple(text_fields or _DEFAULT_TEXT_FIELDS)
    rows = []
    for index, result in enumerate(results):
        signals = _signals(result, fields)
        rows.append({"result_id": result_id(result, index), "signals": signals, "signal_count": len(signals)})
    counts = Counter(signal["type"] for row in rows for signal in row["signals"])
    signal_types = ("metadata_version",) + tuple(signal_type for signal_type, _ in _PATTERNS)
    return {
        "total_results": len(rows),
        "results_with_signals": sum(1 for row in rows if row["signals"]),
        "signal_type_counts": {signal_type: counts.get(signal_type, 0) for signal_type in signal_types},
        "results": rows,
        "warnings": ["no_results"] if not rows else (["no_version_signals"] if not any(row["signals"] for row in rows) else []),
    }


def _signals(result: Any, fields: tuple[str, ...]) -> list[dict[str, str]]:
    found = []
    seen = set()
    for field in _METADATA_FIELDS:
        text = string(value(result, field))
        if text:
            key = ("metadata_version", text.casefold())
            if key not in seen:
                seen.add(key)
                found.append({"type": "metadata_version", "value": text, "field": field})
    body = " ".join(text for field in fields if (text := string(value(result, field))))
    for signal_type, pattern in _PATTERNS:
        for match in pattern.finditer(body):
            value_ = match.group(0)
            key = (signal_type, value_.casefold())
            if key in seen:
                continue
            seen.add(key)
            found.append({"type": signal_type, "value": value_, "field": "text"})
    return found
