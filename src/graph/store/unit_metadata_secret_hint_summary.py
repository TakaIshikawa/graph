"""Summarize likely secret hints in unit metadata."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, metadata, sort_key, unit_id

_SUSPICIOUS_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|bearer)", re.IGNORECASE)
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_BEARER_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._-]{12,}\b", re.IGNORECASE)
_HEX_RE = re.compile(r"\b[a-fA-F0-9]{32,}\b")
_B64_RE = re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b")


def summarize_unit_metadata_secret_hints(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = 0
    hint_counts: Counter[str] = Counter()
    key_path_counts: Counter[str] = Counter()
    affected = set()
    examples = []
    for unit in units:
        total_units += 1
        for key_path, value in _walk(metadata(unit)):
            for hint_type, detected in _hints(key_path, value):
                hint_counts[hint_type] += 1
                key_path_counts[key_path] += 1
                affected.add(unit_id(unit))
                if len(examples) < sample_limit:
                    examples.append({"unit_id": unit_id(unit), "key_path": key_path, "hint_type": hint_type, "redacted_value": _redact(detected)})
    return {
        "total_units": total_units,
        "affected_units": len(affected),
        "hint_type_counts": _counter_rows(hint_counts, "hint_type"),
        "key_path_counts": _counter_rows(key_path_counts, "key_path"),
        "examples": examples,
    }


def _walk(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        return [item for key, child in value.items() for item in _walk(child, f"{prefix}.{field_value(key)}" if prefix else field_value(key))]
    if isinstance(value, list | tuple):
        return [item for index, child in enumerate(value) for item in _walk(child, f"{prefix}[{index}]")]
    return [(prefix, value)] if isinstance(value, str) else []


def _hints(key_path: str, value: str) -> list[tuple[str, str]]:
    hints = []
    suspicious_key = bool(_SUSPICIOUS_KEY_RE.search(key_path))
    if suspicious_key and len(value.strip()) >= 4:
        hints.append(("suspicious_key_path", value))
    for hint_type, pattern in (("sk_prefix", _SK_RE), ("bearer_token", _BEARER_RE), ("long_hex", _HEX_RE), ("base64_like", _B64_RE)):
        for match in pattern.finditer(value):
            hints.append((hint_type, match.group(0)))
    return hints


def _redact(value: str) -> str:
    compact = field_value(value)
    if len(compact) <= 8:
        return "***"
    return f"{compact[:3]}...{compact[-3:]}"


def _counter_rows(counter: Counter[str], key_name: str) -> list[dict[str, Any]]:
    return [{key_name: key, "count": count} for key, count in sorted(counter.items(), key=lambda item: (-item[1], sort_key(item[0])))]
