"""CSV export for high-confidence sensitive token hints in units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "source", "location", "hint_type", "line_number", "redacted_excerpt"]
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_BEARER_RE = re.compile(r"\bBearer\s+([A-Za-z0-9._~+/=-]{12,})", re.IGNORECASE)
_ASSIGNMENT_RE = re.compile(
    r"\b(api[_ -]?key|access[_ -]?token|secret[_ -]?key|client[_ -]?secret|password|token)\b\s*[:=]\s*([^\s,;\"']{6,}|[\"'][^\"']{6,}[\"'])",
    re.IGNORECASE,
)
_SENSITIVE_KEY_RE = re.compile(r"(api[_ -]?key|access[_ -]?token|secret|password|private[_ -]?key|bearer|token)", re.IGNORECASE)


def export_units_to_sensitive_token_hint_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        uid = unit_id(unit)
        source = field_value(get(unit, "source") or metadata(unit).get("source"))
        rows.extend(_content_rows(uid, source, _text(get(unit, "content"))))
        rows.extend(_metadata_rows(uid, source, metadata(unit)))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["location"]), _line_sort(row["line_number"]), sort_key(row["hint_type"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content_rows(unit: str, source: str, content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        hint_type = ""
        if _PRIVATE_KEY_RE.search(line):
            hint_type = "private_key_marker"
        elif _BEARER_RE.search(line):
            hint_type = "bearer_token"
        elif _ASSIGNMENT_RE.search(line):
            hint_type = "secret_assignment"
        if hint_type:
            rows.append(
                {
                    "unit_id": unit,
                    "source": source,
                    "location": "content",
                    "hint_type": hint_type,
                    "line_number": line_number,
                    "redacted_excerpt": _redact(line),
                }
            )
    return rows


def _metadata_rows(unit: str, source: str, meta: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key_path, value in _walk(meta):
        key_name = key_path.rsplit(".", 1)[-1]
        text = field_value(value)
        if _SENSITIVE_KEY_RE.search(key_name):
            rows.append(
                {
                    "unit_id": unit,
                    "source": source,
                    "location": key_path,
                    "hint_type": "metadata_sensitive_key",
                    "line_number": "",
                    "redacted_excerpt": "[REDACTED]" if text else key_name,
                }
            )
        elif isinstance(value, str) and (_BEARER_RE.search(value) or _ASSIGNMENT_RE.search(value) or _PRIVATE_KEY_RE.search(value)):
            rows.append(
                {
                    "unit_id": unit,
                    "source": source,
                    "location": key_path,
                    "hint_type": "metadata_sensitive_value",
                    "line_number": "",
                    "redacted_excerpt": _redact(value),
                }
            )
    return rows


def _walk(value: Any, prefix: str = "metadata") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        return [
            item
            for key in sorted(value, key=sort_key)
            for item in _walk(value[key], f"{prefix}.{field_value(key)}")
        ]
    if isinstance(value, list | tuple):
        return [item for index, child in enumerate(value) for item in _walk(child, f"{prefix}[{index}]")]
    return [(prefix, value)]


def _redact(text: str) -> str:
    redacted = _BEARER_RE.sub("Bearer [REDACTED]", text)
    redacted = _ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)
    return redacted[:120]


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _line_sort(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
